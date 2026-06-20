from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# Third-party (pynput) – installed via requirements.txt

try:
    from pynput import keyboard, mouse
    from pynput.keyboard import Controller as KeyboardController, Key, KeyCode
    from pynput.mouse import Button
except ImportError:
    sys.exit(
        "[ERROR] pynput is not installed.\n"
        "Run:  pip install pynput\n"
        "or:   pip install -r requirements.txt"
    )



# CONFIGURATION

class Sensitivity(Enum):
    VERY_SLOW = "very_slow"
    SLOW      = "slow"
    NORMAL    = "normal"
    FAST      = "fast"


@dataclass
class Config:
    """
    Central configuration.  Adjust these values to tune behaviour.

    transition_delay_ms  – milliseconds to wait between consecutive slot
                           transitions when the queue has multiple items.
                           Smaller = faster; must be > 0.
    sensitivity          – convenience preset that maps to a delay value.
                           Only used if transition_delay_ms is left at 0.
    enable_logging       – print slot-change events to stdout.
    debug_mode           – print additional diagnostic messages.
    """

    sensitivity: Sensitivity = Sensitivity.NORMAL
    transition_delay_ms: float = 0      # 0 means "use sensitivity preset"
    enable_logging: bool = False
    debug_mode: bool = False

    # Delay presets (milliseconds)
    _PRESET_DELAYS: dict = field(default_factory=lambda: {
        Sensitivity.VERY_SLOW: 120,
        Sensitivity.SLOW:       80,
        Sensitivity.NORMAL:     50,
        Sensitivity.FAST:       20,
    }, repr=False)

    def effective_delay_seconds(self) -> float:
        ms = self.transition_delay_ms if self.transition_delay_ms > 0 \
             else self._PRESET_DELAYS[self.sensitivity]
        return ms / 1_000.0



# SLOT MODEL


# Logical ordering of slots
SLOT_ORDER: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
SLOT_INDEX: dict[str, int] = {s: i for i, s in enumerate(SLOT_ORDER)}


class Direction(Enum):
    RIGHT = auto()   # scroll-up / E key  →  higher index
    LEFT  = auto()   # scroll-down / Q key →  lower index



# LOGGING HELPERS


def _setup_logging(debug: bool) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger = logging.getLogger("inventory")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger



# SLOT QUEUE 


class SlotQueue:

    MAX_PENDING: int = 9   # 10 slots − 1 (max useful pending moves)

    def __init__(self) -> None:
        self._q: queue.Queue[Direction] = queue.Queue(maxsize=self.MAX_PENDING)

    def put(self, direction: Direction) -> bool:
        try:
            self._q.put_nowait(direction)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 0.1) -> Optional[Direction]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._q.task_done()

    @property
    def size(self) -> int:
        return self._q.qsize()



# SLOT CONTROLLER  (state machine + key emitter)


class SlotController:

    def __init__(self, cfg: Config, slot_queue: SlotQueue,
                 logger: logging.Logger) -> None:
        self._cfg = cfg
        self._queue = slot_queue
        self._logger = logger

        self._current_slot: str = "1"          # start at slot 1
        self._lock = threading.Lock()           # guards _current_slot
        self._kb = KeyboardController()         # pynput key emitter
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._process_loop,
            name="SlotControllerThread",
            daemon=True,
        )

    
    # Public API
    

    @property
    def current_slot(self) -> str:
        with self._lock:
            return self._current_slot

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    
    # Internal helpers
    

    def _next_slot(self, direction: Direction) -> Optional[str]:

        with self._lock:
            idx = SLOT_INDEX[self._current_slot]

        if direction is Direction.RIGHT:
            new_idx = idx + 1
        else:
            new_idx = idx - 1

        if new_idx < 0 or new_idx >= len(SLOT_ORDER):
            return None                         # boundary – no action

        return SLOT_ORDER[new_idx]

    def _apply_slot(self, new_slot: str) -> None:
        with self._lock:
            self._current_slot = new_slot

        # pynput simulates a physical key press + release
        self._kb.press(KeyCode.from_char(new_slot))
        self._kb.release(KeyCode.from_char(new_slot))

    def _log_transition(self, direction: Direction, old_slot: str,
                        new_slot: Optional[str]) -> None:
        if not self._cfg.enable_logging:
            return
        dir_label = "Scroll Up (←)" if direction is Direction.LEFT \
                    else "Scroll Down (→)"
        if new_slot is None:
            self._logger.info(
                "Current Slot: %s | Input: %s | Action: No Action (boundary)",
                old_slot, dir_label,
            )
        else:
            self._logger.info(
                "Current Slot: %s | Input: %s | New Slot: %s | Pressed Key: %s",
                old_slot, dir_label, new_slot, new_slot,
            )

    
    # Processing loop (daemon thread)
    

    def _process_loop(self) -> None:
        delay = self._cfg.effective_delay_seconds()

        while not self._stop_event.is_set():
            direction = self._queue.get(timeout=0.1)
            if direction is None:
                continue                        # timeout – loop back

            old_slot = self.current_slot
            new_slot = self._next_slot(direction)

            self._log_transition(direction, old_slot, new_slot)

            if new_slot is not None:
                self._apply_slot(new_slot)

            self._queue.task_done()

            # Delay between transitions so rapid scrolling feels smooth
            # and each step is perceptible. We only sleep when there is
            # actually a pending next item or after every processed item
            # to pace output regardless.
            if delay > 0:
                time.sleep(delay)



# INPUT HANDLER  (mouse wheel + Q/E keyboard)


class InputHandler:

    def __init__(self, slot_queue: SlotQueue, logger: logging.Logger,
                 stop_callback) -> None:
        self._queue = slot_queue
        self._logger = logger
        self._stop_callback = stop_callback

        self._mouse_listener: Optional[mouse.Listener] = None
        self._kb_listener: Optional[keyboard.Listener] = None
        self.paused = False

    
    # Mouse callbacks
    

    def _on_scroll(self, x: int, y: int, dx: float, dy: float) -> None:
        if dy == 0:
            return

        direction = Direction.LEFT if dy > 0 else Direction.RIGHT
        accepted = self._queue.put(direction)

        if self._logger.isEnabledFor(logging.DEBUG):
            status = "queued" if accepted else "dropped (queue full)"
            self._logger.debug("Mouse scroll dy=%+.1f → %s [%s]",
                               dy, direction.name, status)

    
    # Keyboard callbacks
    

    def _on_key_press(self, key) -> None:
        try:
            char = key.char
        except AttributeError:
            char = None

        if char == "q" or char == "Q":
            accepted = self._queue.put(Direction.LEFT)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("Key Q pressed → LEFT [%s]",
                                   "queued" if accepted else "dropped")

        elif char == "e" or char == "E":
            accepted = self._queue.put(Direction.RIGHT)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("Key E pressed → RIGHT [%s]",
                                   "queued" if accepted else "dropped")

        elif key == Key.esc:
            self.paused = not self.paused

            if self.paused:
                self._logger.info("PAUSED")
            else:
                self._logger.info("RESUMED")

    
    # Lifecycle
    

    def start(self) -> None:
        try:
            self._mouse_listener = mouse.Listener(on_scroll=self._on_scroll)
            self._mouse_listener.start()
        except Exception as exc:
            raise RuntimeError(f"Failed to hook mouse: {exc}") from exc

        try:
            self._kb_listener = keyboard.Listener(on_press=self._on_key_press)
            self._kb_listener.start()
        except Exception as exc:
            if self._mouse_listener:
                self._mouse_listener.stop()
            raise RuntimeError(f"Failed to hook keyboard: {exc}") from exc

    def stop(self) -> None:
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._kb_listener:
            self._kb_listener.stop()



# APPLICATION ENTRY POINT


def main() -> None:
    
    # Config – edit here or load from a file/CLI in a real application
    
    cfg = Config(
        sensitivity=Sensitivity.NORMAL,
        transition_delay_ms=0,      # 0 → use sensitivity preset
        enable_logging=False,
        debug_mode=False,
    )

    logger = _setup_logging(cfg.debug_mode)
    logger.info("=" * 60)
    logger.info("Inventory Slot Controller")
    logger.info("Slots : 1 2 3 4 5 6 7 8 9 0")
    logger.info("Scroll Down / E  →  move right")
    logger.info("Scroll Up / Q →  move left")
    logger.info("Press Esc to exit.")
    logger.info("=" * 60)
    logger.info("Sensitivity : %s  (delay %d ms)",
                cfg.sensitivity.value,
                int(cfg.effective_delay_seconds() * 1000))
    logger.info("Starting at slot : 1")
    logger.info("-" * 60)

    
    # Wire up components
    
    slot_queue  = SlotQueue()
    stop_event  = threading.Event()
    controller  = SlotController(cfg, slot_queue, logger)

    def request_stop() -> None:
        stop_event.set()

    input_handler = InputHandler(slot_queue, logger, stop_callback=request_stop)

    
    # Start
    
    try:
        controller.start()
        input_handler.start()
    except RuntimeError as exc:
        logger.error("Startup failed: %s", exc)
        logger.error("Ensure you have the necessary OS permissions to hook "
                     "input devices (e.g. run as administrator on Windows, "
                     "or grant Accessibility access on macOS).")
        sys.exit(1)

    
    # Main thread: wait for stop signal (Esc or Ctrl+C)
    
    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received – shutting down.")
    finally:
        input_handler.stop()
        controller.stop()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
