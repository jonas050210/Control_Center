"""Construct and drive the desktop Control Center without Tk or a display.

The Tk suite (``python/tests/test_control_center_desktop.py``) is the real
check: it builds genuine widgets and runs in the ``desktop-ui-tests`` CI
job. This script exists for the other environment - a headless machine
without ``python3-tk`` - where that suite *skips* and a rewrite of the GUI
would otherwise never be executed at all before it reaches a user.

It installs a deliberately small fake ``tkinter`` (geometry is bookkeeping,
``after`` callbacks never fire, unknown widget methods are no-ops) and then
builds the real application, shows and refreshes every page, cycles every
theme, density, motion level and shell layout, drives the Settings layout
studio and presets, and exercises the Training and Benchmark plan logic.

What it does **not** verify: pixels, colours, fonts, DPI, real geometry,
real event dispatch, or that a native Tk build accepts every option. A pass
here means "no name errors, no bad wiring, no constructor crashes", nothing
more. Do not cite it as GUI test coverage.

    python3 tools/control_center_smoke.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import traceback
import types
from collections.abc import Callable

_AFTER_IDS = iter(range(1, 100000))


class TclError(Exception):
    pass


class Event:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _noop(*_args, **_kw):
    return None


class _Base:
    def __init__(self, master=None, **kw):
        object.__setattr__(self, "_tk_options", dict(kw))
        object.__setattr__(self, "_tk_children", [])
        object.__setattr__(self, "_tk_manager", None)
        object.__setattr__(self, "_tk_destroyed", False)
        object.__setattr__(self, "_tk_binds", {})
        object.__setattr__(self, "_tk_master", master if isinstance(master, _Base) else None)
        object.__setattr__(self, "tk", master.tk if isinstance(master, _Base) else self)
        if isinstance(master, _Base):
            master._tk_children.append(self)

    # -- options ---------------------------------------------------------
    def configure(self, cnf=None, **kw):
        if isinstance(cnf, dict):
            self._tk_options.update(cnf)
        self._tk_options.update(kw)
        return None

    config = configure

    def cget(self, key):
        return self._tk_options.get(str(key), "")

    def __getitem__(self, key):
        return self.cget(key)

    def __setitem__(self, key, value):
        self._tk_options[str(key)] = value

    def keys(self):
        return list(self._tk_options)

    # -- geometry --------------------------------------------------------
    def _manage(self, manager, kw):
        object.__setattr__(self, "_tk_manager", manager)
        return self

    def pack(self, **kw):
        return self._manage("pack", kw)

    def grid(self, **kw):
        return self._manage("grid", kw)

    def place(self, **kw):
        return self._manage("place", kw)

    def pack_forget(self):
        return self._manage(None, {})

    grid_forget = pack_forget
    place_forget = pack_forget
    grid_remove = pack_forget
    pack_propagate = _noop
    grid_propagate = _noop

    def pack_configure(self, **kw):
        return self._manage("pack", kw)

    def grid_configure(self, **kw):
        return self._manage("grid", kw)

    def place_configure(self, **kw):
        return self._manage("place", kw)

    def pack_info(self, **kw):
        return {}

    def grid_info(self, **kw):
        return {}

    def place_info(self, **kw):
        return {}

    def columnconfigure(self, *a, **kw):
        return None

    def rowconfigure(self, *a, **kw):
        return None

    grid_columnconfigure = columnconfigure
    grid_rowconfigure = rowconfigure

    # -- introspection ---------------------------------------------------
    def winfo_children(self):
        return list(self._tk_children)

    def winfo_manager(self):
        return self._tk_manager or ""

    def winfo_exists(self):
        return 0 if self._tk_destroyed else 1

    def winfo_width(self):
        return 200

    def winfo_height(self):
        return 100

    def winfo_reqwidth(self):
        return 200

    def winfo_reqheight(self):
        return 100

    def winfo_x(self):
        return 0

    winfo_y = winfo_x
    winfo_rootx = winfo_x
    winfo_rooty = winfo_x

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def winfo_viewable(self):
        return 1

    def winfo_ismapped(self):
        return 1

    def winfo_toplevel(self):
        node = self
        while isinstance(getattr(node, "_tk_master", None), _Base):
            node = node._tk_master
        return node

    def winfo_name(self):
        return f"w{id(self) % 10000}"

    # -- events ----------------------------------------------------------
    def bind(self, sequence=None, func=None, add=None):
        if sequence is not None and func is not None:
            self._tk_binds.setdefault(sequence, []).append(func)
        return f"bind-{sequence}"

    def unbind(self, sequence, funcid=None):
        return None

    bind_all = bind
    bind_class = bind
    tag_bind = bind

    def event_generate(self, sequence, **kw):
        for func in self._tk_binds.get(sequence, []):
            func(Event(**kw))

    # -- timers ----------------------------------------------------------
    def after(self, ms, func=None, *args):
        return next(_AFTER_IDS)

    def after_cancel(self, identifier=None):
        return None

    after_idle = after

    # -- lifecycle -------------------------------------------------------
    def update(self):
        return None

    update_idletasks = update
    mainloop = update
    quit = update
    wait_window = update

    def destroy(self):
        object.__setattr__(self, "_tk_destroyed", True)
        for child in list(self._tk_children):
            child.destroy()
        self._tk_children.clear()

    def focus_set(self):
        return None

    focus_force = focus_set

    def focus_get(self):
        return self

    def grab_set(self):
        return None

    def grab_release(self):
        return None

    def lift(self, *a):
        return None

    def lower(self, *a):
        return None

    def bell(self):
        return None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _noop


class _TkApp(_Base):
    def __init__(self):
        super().__init__(None)
        self._tk_children = []

    def title(self, *a, **kw):
        return None

    def geometry(self, spec=None):
        return "1720x1000+0+0"

    def minsize(self, *a, **kw):
        return None

    maxsize = minsize
    resizable = minsize
    iconphoto = minsize
    iconbitmap = minsize
    protocol = minsize
    transient = minsize
    attributes = minsize
    wm_attributes = minsize
    wm_title = minsize
    class_ = minsize

    def state(self, *a):
        return "normal"

    def option_add(self, *a, **kw):
        return None

    def clipboard_clear(self):
        return None


class Variable:
    def __init__(self, master=None, value=None, name=None):
        self._value = value
        self._callbacks = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for callback in tuple(self._callbacks):
            callback("", "", "write")

    def trace_add(self, mode, callback):
        self._callbacks.append(callback)
        return f"trace-{len(self._callbacks)}"

    def trace_remove(self, *a):
        return None

    trace = trace_add


class StringVar(Variable):
    def __init__(self, master=None, value="", name=None):
        super().__init__(master, value, name)


class IntVar(Variable):
    def __init__(self, master=None, value=0, name=None):
        super().__init__(master, value, name)

    def get(self):
        try:
            return int(self._value)
        except (TypeError, ValueError):
            return 0


class BooleanVar(Variable):
    def __init__(self, master=None, value=False, name=None):
        super().__init__(master, value, name)

    def get(self):
        return bool(self._value)


class DoubleVar(Variable):
    def __init__(self, master=None, value=0.0, name=None):
        super().__init__(master, value, name)

    def get(self):
        try:
            return float(self._value)
        except (TypeError, ValueError):
            return 0.0


class Misc:
    pass


class Widget(_Base):
    pass


class Tk(_TkApp):
    pass


class _Entry(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._text = str(kw.get("text", "") or kw.get("textvariable") or "")

    def insert(self, index, text):
        self._text = str(self._text) + str(text)

    def delete(self, first, last=None):
        self._text = "nonsense"

    def get(self):
        return str(self._text)


class Text(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._content = ""

    def insert(self, index, chars, *tags):
        self._content += str(chars)

    def delete(self, first, last=None):
        self._content = ""

    def get(self, first, last=None):
        return self._content

    def index(self, spec):
        return "1.0"

    def see(self, *a):
        return None

    def yview(self, *a):
        return (0.0, 1.0)

    def xview(self, *a):
        return (0.0, 1.0)

    def _accepts(self, *a, **kw):
        return None

    yview_moveto = _accepts
    yview_scroll = _accepts
    xview_moveto = _accepts
    xview_scroll = _accepts
    mark_set = _accepts
    tag_configure = _accepts
    tag_add = _accepts
    tag_remove = _accepts
    edit_undo = _accepts
    edit_redo = _accepts
    edit_reset = _accepts
    window_create = _accepts

    def tag_ranges(self, *a):
        return ()

    def compare(self, *a):
        return True

    def search(self, *a):
        return ""

    def count(self, *a):
        return (0, 0)


class Canvas(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._items = {}
        self._next = 1

    def _new(self, kind, **kw):
        item = self._next
        self._next += 1
        self._items[item] = (kind, kw)
        return item

    def create_rectangle(self, *a, **kw):
        return self._new("rectangle", **kw)

    create_oval = create_rectangle
    create_polygon = create_rectangle
    create_line = create_rectangle
    create_arc = create_rectangle
    create_window = create_rectangle
    create_image = create_rectangle

    def create_text(self, *a, **kw):
        return self._new("text", **kw)

    def delete(self, *a):
        return None

    def itemconfigure(self, *a, **kw):
        return None

    # -- scrolling --------------------------------------------------------
    # Real canvases always answer yview() with a (first, last) pair; the
    # restore path in Page._restore_view_state reads it, so the double must
    # not answer None there (Tk never does).
    def yview(self, *a):
        if a and a[0] == "moveto":
            self._tk_options["yview"] = a[1]
        return (0.0, 1.0)

    def xview(self, *a):
        return (0.0, 1.0)

    def yview_moveto(self, fraction):
        self._tk_options["yview"] = fraction
        return None

    def configure_scrollregion(self, *a):
        return None

    itemconfig = itemconfigure

    def itemcget(self, *a, **kw):
        return ""

    def coords(self, *a, **kw):
        return (0, 0, 0, 0)

    def bbox(self, *a, **kw):
        return (0, 0, 10, 10)

    def move(self, *a, **kw):
        return None

    def scale(self, *a, **kw):
        return None

    def find_all(self, *a, **kw):
        return ()

    find_withtag = find_all

    def tag_raise(self, *a, **kw):
        return None

    tag_lower = tag_raise


class Listbox(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._items = []

    def insert(self, index, *elements):
        self._items.extend(elements)

    def delete(self, first, last=None):
        self._items.clear()

    def get(self, first, last=None):
        return tuple(self._items)

    def size(self):
        return len(self._items)

    def curselection(self):
        return ()

    def see(self, *a):
        return None

    def activate(self, *a):
        return None

    def selection_clear(self, *a):
        return None


class Scale(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._value = kw.get("variable")
        self._command = kw.get("command")

    def set(self, value):
        if isinstance(self._value, Variable):
            self._value.set(value)
        if callable(self._command):
            self._command(value)

    def get(self):
        return self._value.get() if isinstance(self._value, Variable) else 0


class Toplevel(Tk):
    def __init__(self, master=None, **kw):
        super().__init__(**kw)


class PhotoImage(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)

    def width(self):
        return 16

    def height(self):
        return 16

    def put(self, *a, **kw):
        return None

    def get(self, *a):
        return 0

    def sub_sample(self, *a):
        return self

    def zoom(self, *a):
        return self

    def copy(self):
        return self


class Font(Widget):
    def __init__(self, root=None, **kw):
        super().__init__(None, **kw)

    def measure(self, text):
        return len(str(text)) * 7

    def metrics(self, *a):
        return 16

    def actual(self, *a, **kw):
        return self._tk_options.get("size", 12)

    def configure(self, cnf=None, **kw):
        return super().configure(cnf, **kw)

    def cget(self, key):
        return self._tk_options.get(str(key), "")


class _Treeview(Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._rows = {}
        self._columns = kw.get("columns", ())
        self._selection = ()
        self._children = {}

    def insert(self, parent, index, iid=None, **kw):
        iid = iid or f"row{len(self._rows)}"
        self._rows[iid] = kw
        self._children.setdefault(str(parent), []).append(iid)
        return iid

    def delete(self, *items):
        for item in items:
            self._rows.pop(item, None)

    def get_children(self, item=""):
        return tuple(self._children.get(str(item), ()))

    def parent(self, item):
        return ""

    def item(self, item, option=None, **kw):
        if kw:
            self._rows.setdefault(item, {}).update(kw)
        if option:
            return self._rows.get(item, {}).get(option, "")
        return self._rows.get(item, {})

    def heading(self, *a, **kw):
        return None

    def column(self, *a, **kw):
        return None

    def set(self, item, column=None, value=None):
        row = self._rows.setdefault(item, {})
        if column is None:
            row.update(value or {})
        else:
            row[column] = value

    def selection(self, *items):
        if items:
            self._selection = tuple(items)
            return None
        return self._selection

    def selection_set(self, *items):
        self._selection = tuple(items)

    def selection_remove(self, *items):
        self._selection = ()

    def selection_add(self, *items):
        self._selection = tuple(items)

    def see(self, *a):
        return None

    def tag_configure(self, *a, **kw):
        return None

    tag_add = tag_configure

    def tag_has(self, *a, **kw):
        return ()

    tag_bind = tag_configure

    def exists(self, item):
        return item in self._rows

    def identify_region(self, *a):
        return "cell"

    def identify_column(self, *a):
        return "#1"

    def identify_row(self, *a):
        return ""

    def bbox(self, *a, **kw):
        return (0, 0, 100, 20)

    def xview(self, *a):
        return (0.0, 1.0)

    def yview(self, *a):
        return (0.0, 1.0)

    def xview_moveto(self, *a, **kw):
        return None

    yview_moveto = xview_moveto
    xview_scroll = xview_moveto
    yview_scroll = xview_moveto

    def move(self, *a):
        return None


class Scrollbar(Widget):
    """A dumb native scrollbar: ``set(first, last)`` plus a command."""

    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._command = kw.get("command")
        self._range = (0.0, 1.0)

    def set(self, first, last):
        self._range = (float(first), float(last))

    def get(self):
        return self._range


class _Notebook(Widget):
    def add(self, child, **kw):
        return None

    def select(self, tab=None):
        return tab

    def tabs(self):
        return ()

    def index(self, tab):
        return 0

    def enable_traversal(self, *a):
        return None


class _Combobox(_Entry):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self._values = tuple(kw.get("values", ()))

    def configure(self, cnf=None, **kw):
        if "values" in kw:
            self._values = tuple(kw["values"])
        return super().configure(cnf, **kw)

    def cget(self, key):
        if str(key) == "values":
            return self._values
        return super().cget(key)

    def set(self, value):
        var = self._tk_options.get("textvariable")
        if isinstance(var, Variable):
            var.set(value)
        self._text = str(value)

    def current(self, index=None):
        if index is None:
            return 0
        if isinstance(self._tk_options.get("textvariable"), Variable):
            self._tk_options["textvariable"].set(self._values[index % len(self._values)])


class _Style:
    def __init__(self, master=None):
        self._settings = {}

    def configure(self, style, **kw):
        self._settings.setdefault(style, {}).update(kw)

    def map(self, style, **kw):
        self._settings.setdefault(style, {}).update(kw)

    def lookup(self, style, option, default=""):
        return self._settings.get(style, {}).get(option, default)

    def theme_use(self, name=None):
        return "clam"

    def theme_names(self):
        return ("clam", "alt", "default")

    def layout(self, style, spec=None):
        return []

    def element_create(self, *a, **kw):
        return None

    def element_names(self):
        return ()


class _TTKModule(types.ModuleType):
    def __init__(self):
        super().__init__("tkinter.ttk")
        self.style = _Style()
        self.Style = _Style
        for name in (
            "Frame",
            "Label",
            "Button",
            "Entry",
            "Checkbutton",
            "Radiobutton",
            "Combobox",
            "Separator",
            "Progressbar",
            "LabelFrame",
            "Notebook",
            "PanedWindow",
            "Scrollbar",
            "Spinbox",
            "Sizegrip",
            "Menubutton",
            "Scale",
        ):
            setattr(self, name, type(name, (Widget,), {}))
        # The behavioural doubles: a page's tables and drop-downs have to
        # behave like the real widget (rows, selections, values), because the
        # handlers under test read them back.
        self.Treeview = _Treeview
        self.Combobox = _Combobox
        # Entry/Text are behavioural too: handlers read ``get()`` back and a
        # generic double would return ``None``, which then looks like a
        # product bug instead of a stub gap.
        self.Entry = _Entry
        self.Text = Text
        self.Notebook = _Notebook
        self.Scrollbar = Scrollbar

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (Widget,), {})
        setattr(self, name, cls)
        return cls


class _StyleShim:
    styles = ()

    def __call__(self, *a, **kw):
        return None


class _TextModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _noop


def _make_module():
    module = types.ModuleType("tkinter")
    module.TclError = TclError
    module.Event = Event
    module.Variable = Variable
    module.StringVar = StringVar
    module.IntVar = IntVar
    module.BooleanVar = BooleanVar
    module.DoubleVar = DoubleVar
    module.Tk = Tk
    module.Toplevel = Toplevel
    module.Widget = Widget
    module.Misc = Misc
    module.Frame = type("Frame", (Widget,), {})
    module.Label = type("Label", (Widget,), {})
    module.Button = type("Button", (_Entry,), {})
    module.Entry = _Entry
    module.Text = Text
    module.Canvas = Canvas
    module.Listbox = Listbox
    module.Scale = Scale
    module.PhotoImage = PhotoImage
    module.Menu = type("Menu", (Widget,), {})
    module.Checkbutton = type("Checkbutton", (Widget,), {})
    module.Radiobutton = type("Radiobutton", (Widget,), {})
    module.Message = type("Message", (Widget,), {})
    module.font = _TextModule("tkinter.font")
    module.font.Font = Font

    def families(*a):
        return ("Segoe UI", "Consolas")

    def nametofont(name):
        return Font()

    module.font.families = families
    module.font.nametofont = nametofont
    module.font.fixed = "Consolas"
    module.messagebox = _TextModule("tkinter.messagebox")
    module.messagebox.askyesno = lambda *a, **kw: True
    module.messagebox.askokcancel = lambda *a, **kw: True
    module.filedialog = _TextModule("tkinter.filedialog")
    module.filedialog.askopenfilename = lambda *a, **kw: ""
    module.filedialog.asksaveasfilename = lambda *a, **kw: ""
    module.filedialog.askdirectory = lambda *a, **kw: ""
    module.ttk = _StyleShim()
    for name in (
        "END",
        "LEFT",
        "RIGHT",
        "TOP",
        "BOTTOM",
        "BOTH",
        "X",
        "Y",
        "N",
        "S",
        "E",
        "W",
        "NSEW",
        "CENTER",
        "WORD",
        "NONE",
        "DISABLED",
        "NORMAL",
        "ACTIVE",
        "HORIZONTAL",
        "VERTICAL",
        "SUNKEN",
        "RAISED",
        "FLAT",
        "GROOVE",
        "RIDGE",
        "INSERT",
        "SEL",
        "SEL_FIRST",
        "SEL_LAST",
        "TRUE",
        "FALSE",
        "ANCHOR",
        "NW",
        "NE",
        "SW",
        "SE",
        "EW",
        "NS",
        "w",
        "e",
        "n",
        "s",
        "nw",
        "ne",
        "sw",
        "se",
    ):
        setattr(module, name, name)
    module.constants = _TextModule("tkinter.constants")
    for name in dir(module):
        setattr(module.constants, name, getattr(module, name))

    def _unknown(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (Widget,), {})
        setattr(self, name, cls)
        return cls

    module.__class__ = type("_TkInterModule", (types.ModuleType,), {"__getattr__": _unknown})
    return module


def install_fake_tkinter() -> types.ModuleType:
    """Install the fake ``tkinter`` (and friends) into ``sys.modules``.

    Returns the fake top-level module. Import order matters: this must run
    before anything imports ``tkinter``.
    """
    import sys

    if "tkinter" in sys.modules and not isinstance(sys.modules["tkinter"], types.ModuleType):
        return sys.modules["tkinter"]
    module = _make_module()
    ttk = _TTKModule()
    module.ttk = ttk
    sys.modules["tkinter"] = module
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.font"] = module.font
    sys.modules["tkinter.messagebox"] = module.messagebox
    sys.modules["tkinter.filedialog"] = module.filedialog
    sys.modules["tkinter.constants"] = module.constants
    return module


# ---------------------------------------------------------------------------
# The smoke run
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


#: Callback exceptions the runner swallowed, so a passing smoke run cannot
#: hide a page handler that raises: ``BackgroundRunner._pump`` deliberately
#: prints and continues, which is right for a live window and invisible to a
#: script. The recorder is installed once per run.
_CALLBACK_ERRORS: list[BaseException] = []
_CALLBACK_TRACES: list[str] = []


def _capture_callback_errors() -> None:
    """Record (and still print) every exception the polling loop swallows."""
    import traceback as traceback_module

    original = traceback_module.print_exc

    def record(*args: object, **kwargs: object) -> None:
        error = sys.exc_info()[1]
        trace = sys.exc_info()[2]
        frame = trace
        from_pump = False
        while frame is not None:
            if frame.tb_frame.f_code.co_name == "_pump":
                from_pump = True
                break
            frame = frame.tb_next
        # Keep the full formatted traceback: a background callback runs on
        # nobody's stack, so without it the only clue is the exception class.
        # Step failures go through the same patched ``print_exc``, so only
        # record what actually came out of the polling loop.
        if from_pump:
            _CALLBACK_ERRORS.append(error or RuntimeError("background callback failed"))
            _CALLBACK_TRACES.append(traceback_module.format_exc())
        original(*args, **kwargs)

    traceback_module.print_exc = record  # type: ignore[assignment]


def _drain(app: object, attempts: int = 6, delay: float = 0.03) -> None:
    """Let real background work finish and run its completion callbacks.

    Page refreshes are genuinely asynchronous (a thread pool hands results
    back through a queue that a Tk timer drains). Running that queue here is
    what exercises the ``_on_*`` handlers, so a wrong callback signature or
    a handler that trips over an empty project fails loudly instead of
    hiding behind an unrun poll.
    """
    import time

    for _ in range(attempts):
        app.background._pump()  # type: ignore[attr-defined]
        app.update()  # type: ignore[attr-defined]
        time.sleep(delay)


def _step(failures: list[str], name: str, fn: Callable[[], object]) -> None:
    try:
        fn()
        print(f"  ok    {name}")
    except Exception:  # noqa: BLE001 - every failure is reported, none abort the run
        failures.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc(limit=6)


def _exercise_shell(app: object, page_classes: tuple[type, ...]) -> None:
    for mode in ("topbar", "board", "rail"):
        app.set_layout_mode(mode)  # type: ignore[attr-defined]
        for page_class in page_classes:
            app.show_page(page_class.title)  # type: ignore[attr-defined]


def _exercise_studio(page: object, page_widgets: dict) -> None:
    for title, specs in page_widgets.items():
        page.studio_page_var.set(title)  # type: ignore[attr-defined]
        page._render_studio_rows()  # type: ignore[attr-defined]
        if not specs:
            continue
        first = specs[0].widget_id
        page._move_studio_widget(first, 1)  # type: ignore[attr-defined]
        page._move_studio_widget(first, -1)  # type: ignore[attr-defined]
        page._span_studio_widget(first, min(2, specs[0].max_span))  # type: ignore[attr-defined]
        page._toggle_studio_widget(first, False)  # type: ignore[attr-defined]
        page._toggle_studio_widget(first, True)  # type: ignore[attr-defined]
    page._reset_studio_page()  # type: ignore[attr-defined]
    page._reset_studio_all()  # type: ignore[attr-defined]


def _exercise_presets(page: object) -> None:
    page.preset_name_var.set("smoke test")  # type: ignore[attr-defined]
    page._save_preset()  # type: ignore[attr-defined]
    page._refresh_preset_list()  # type: ignore[attr-defined]
    page.preset_choice_var.set("smoke test")  # type: ignore[attr-defined]
    page._apply_preset()  # type: ignore[attr-defined]
    page.preset_name_var.set("smoke test 2")  # type: ignore[attr-defined]
    page._rename_preset()  # type: ignore[attr-defined]
    page.preset_choice_var.set("smoke test 2")  # type: ignore[attr-defined]
    page._apply_preset()  # type: ignore[attr-defined]
    # Renaming onto a name that is already taken must be refused, not
    # silently destroy the other preset (the store raises, the page notify
    # path handles it, and nothing crashes).
    page.preset_name_var.set("smoke clash")  # type: ignore[attr-defined]
    page._save_preset()  # type: ignore[attr-defined]
    page.preset_choice_var.set("smoke test 2")  # type: ignore[attr-defined]
    page.preset_name_var.set("smoke clash")  # type: ignore[attr-defined]
    page._rename_preset()  # type: ignore[attr-defined]
    page.preset_choice_var.set("smoke clash")  # type: ignore[attr-defined]
    page._delete_preset()  # type: ignore[attr-defined]
    page.preset_choice_var.set("smoke test 2")  # type: ignore[attr-defined]
    page._delete_preset()  # type: ignore[attr-defined]


def _exercise_accent(app: object) -> None:
    """Pick a curated accent, type a hex value, then fall back to the theme."""
    page = app.pages["Settings"]
    page._pick_accent("#F5A524")  # type: ignore[attr-defined]
    if app.palette.accent.lower() != "#f5a524":
        raise AssertionError(f"a picked accent did not reach the palette: {app.palette.accent}")
    if app.prefs.accent.lower() != "#f5a524":
        raise AssertionError("a picked accent was not remembered in the preferences")
    page.accent_var.set("#22D3EE")
    page._apply_accent()  # type: ignore[attr-defined]
    if app.palette.accent.lower() != "#22d3ee":
        raise AssertionError("a typed accent was not applied")
    page.accent_var.set("not a colour")
    page._apply_accent()  # type: ignore[attr-defined]
    if app.prefs.accent.lower() != "#22d3ee":
        raise AssertionError("an unusable accent must be refused, not stored")
    page._clear_accent()  # type: ignore[attr-defined]
    if app.prefs.accent:
        raise AssertionError("clearing the accent must return to the theme's own")
    if page.accent_hint.cget("text") and "theme" not in str(page.accent_hint.cget("text")).lower():
        raise AssertionError("the accent hint must say which accent is active")


def _exercise_preset_transfer(app: object) -> None:
    """Export a preset to a file, import it back under a new name."""
    import json
    import tempfile
    from pathlib import Path

    page = app.pages["Settings"]
    page.preset_name_var.set("smoke export")  # type: ignore[attr-defined]
    page._save_preset()  # type: ignore[attr-defined]
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "exported.json"
        if not app.export_layout_preset("smoke export", target):
            raise AssertionError("export_layout_preset reported failure")
        if not target.is_file():
            raise AssertionError("export_layout_preset wrote no file")
        document = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(document.get("layout"), dict):
            raise AssertionError("an exported preset must carry its layout")
        document["name"] = "smoke imported"
        target.write_text(json.dumps(document), encoding="utf-8")
        imported = app.import_layout_preset(target)
        if imported != "smoke imported":
            raise AssertionError(f"import did not produce the expected name: {imported!r}")
        if "smoke imported" not in app.list_layout_presets():
            raise AssertionError("an imported preset must appear in the preset list")
        # A clash is refused unless overwriting is asked for.
        if app.import_layout_preset(target) is not None:
            raise AssertionError("importing onto an existing name must be refused")
        if app.import_layout_preset(target, overwrite=True) != "smoke imported":
            raise AssertionError("importing with overwrite=True must succeed")
        if app.preset_name_for_import(target) != "smoke imported":
            raise AssertionError("preset_name_for_import must report the target name")
    # A preset carries its appearance: applying it must restore the accent,
    # the theme and the density, through the same paths the pickers use.
    app.set_accent("#F5A524")
    app.set_theme("lime")
    app.set_density("compact")
    page.preset_name_var.set("smoke appearance")  # type: ignore[attr-defined]
    page._save_preset()  # type: ignore[attr-defined]
    app.set_accent("")
    app.set_theme("corz")
    app.set_density("comfort")
    if not app.apply_layout_preset("smoke appearance"):
        raise AssertionError("a saved preset could not be applied again")
    if app.prefs.accent.lower() != "#f5a524" or app.palette.accent.lower() != "#f5a524":
        raise AssertionError(f"a preset must restore its accent, got {app.prefs.accent!r}")
    if app.bus.theme.name != "lime" or app.prefs.theme != "lime":
        raise AssertionError(f"a preset must restore its theme, got {app.bus.theme.name!r}")
    if app.prefs.density != "compact" or app.bus.density.name != "compact":
        raise AssertionError(f"a preset must restore its density, got {app.prefs.density!r}")
    app.set_accent("")
    app.set_theme("corz")
    app.set_density("comfort")
    for name in ("smoke export", "smoke imported", "smoke appearance"):
        app.pages["Settings"].preset_choice_var.set(name)  # type: ignore[attr-defined]
        app.pages["Settings"]._delete_preset()  # type: ignore[attr-defined]


def _exercise_appearance(page: object) -> None:
    page.theme_var.set("Neon Lime")  # type: ignore[attr-defined]
    page._on_theme_selected()  # type: ignore[attr-defined]
    page.layout_var.set("Top bar")  # type: ignore[attr-defined]
    page._on_layout_selected()  # type: ignore[attr-defined]
    page.density_var.set("Compact")  # type: ignore[attr-defined]
    page._on_density_selected()  # type: ignore[attr-defined]
    page.motion_var.set("Reduced")  # type: ignore[attr-defined]
    page._on_motion_selected()  # type: ignore[attr-defined]
    page.radius_var.set(14)  # type: ignore[attr-defined]
    page._on_radius_changed()  # type: ignore[attr-defined]
    page._on_effect_toggled()  # type: ignore[attr-defined]
    page._reset_appearance()  # type: ignore[attr-defined]


def _exercise_training(page: object) -> None:
    page.budget_mode.set("time")  # type: ignore[attr-defined]
    page._on_budget_mode("time")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    page.budget_minutes_var.set("12")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    page.budget_mode.set("steps")  # type: ignore[attr-defined]
    page._on_budget_mode("steps")  # type: ignore[attr-defined]
    page._apply_step_preset("25000")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    page.field_vars["environment_count"].set("8")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    page.field_vars["environment_count"].set("oops")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    page.field_vars["environment_count"].set("8")  # type: ignore[attr-defined]


def _exercise_benchmarks(page: object) -> None:
    for mode in ("custom", "push", "auto"):
        page.mode_control.set(mode)  # type: ignore[attr-defined]
        page._on_mode_changed(mode)  # type: ignore[attr-defined]
        page.refresh()  # type: ignore[attr-defined]
    page.custom_env_var.set("16,32")  # type: ignore[attr-defined]
    page.custom_worker_var.set("4")  # type: ignore[attr-defined]
    page.custom_steps_var.set("50000")  # type: ignore[attr-defined]
    page.mode_control.set("custom")  # type: ignore[attr-defined]
    page._on_mode_changed("custom")  # type: ignore[attr-defined]
    page.refresh()  # type: ignore[attr-defined]
    if page.current_plan()["errors"]:  # type: ignore[attr-defined]
        raise AssertionError(f"valid custom plan reported errors: {page.current_plan()}")  # type: ignore[attr-defined]
    if str(page.run_button.cget("state")) == "disabled":  # type: ignore[attr-defined]
        raise AssertionError("a valid custom plan must leave the run button enabled")
    page.custom_worker_var.set("nonsense")  # type: ignore[attr-defined]
    if not page.current_plan()["errors"]:  # type: ignore[attr-defined]
        raise AssertionError("a malformed worker list must be reported as a plan error")
    if str(page.run_button.cget("state")) != "disabled":  # type: ignore[attr-defined]
        raise AssertionError("a malformed custom plan must disable the run button")
    page.custom_worker_var.set("4")  # type: ignore[attr-defined]
    if str(page.run_button.cget("state")) == "disabled":  # type: ignore[attr-defined]
        raise AssertionError("fixing the list must make the run button usable again")


def _exercise_widgets(app: object) -> None:
    """Build and drive the reusable widgets on a scratch frame."""
    import tkinter as tk

    from sandboxai.control_center_ui import (
        AnimatedValue,
        SegmentedControl,
        StatusDot,
        ToastHost,
    )
    from sandboxai.control_center_widgets import (
        LineChart,
        LogPanel,
        PhaseStepper,
        StatRow,
        ToolTip,
        _scrollable_table,
    )

    host = tk.Frame(app)  # type: ignore[attr-defined]
    host.pack()

    row = StatRow(host, ("A", "B"), bus=app.bus, motion=app.motion)  # type: ignore[attr-defined]
    row.update_values({"A": ("41.5", None), "B": ("7", app.palette.ok)})
    row.update_values(
        {"A": ("42.0", None)},
        numeric={"A": (42.0, "n/a", lambda value: f"{value:.1f}")},
    )
    app.motion._tick()  # type: ignore[attr-defined]
    app.motion._tick()  # type: ignore[attr-defined]
    app.motion.stop_all()  # type: ignore[attr-defined]

    stepper = PhaseStepper(host, bus=app.bus, motion=app.motion)  # type: ignore[attr-defined]
    stepper.set_state(
        [{"name": "plan", "status": "done"}, {"name": "measure", "status": "active"}], 0.4
    )
    stepper.set_phases([{"name": "one", "status": "pending"}], 0.0)
    stepper._on_anim_tick()  # type: ignore[attr-defined]

    chart = LineChart(host, "throughput", bus=app.bus)  # type: ignore[attr-defined]
    chart.set_points([(1.0, 10.0), (2.0, 21.0), (3.0, 19.0)])
    chart.point_count()
    chart._on_motion(Event(x=40, y=25))  # type: ignore[attr-defined]
    chart._on_leave(Event())  # type: ignore[attr-defined]
    chart.set_points([])

    panel = LogPanel(host, max_lines=12, bus=app.bus, motion=app.motion)  # type: ignore[attr-defined]
    panel.apply_log({"stdout": ["hello", "world"], "stderr": ["warn"]})
    panel.apply_log({"stdout": [], "stderr": [], "exited": True, "returncode": 3})
    panel._on_wrap_toggled()  # type: ignore[attr-defined]
    panel._on_autoscroll_toggled()  # type: ignore[attr-defined]
    panel._on_manual_scroll()  # type: ignore[attr-defined]
    panel._on_text_scroll("0.0", "1.0")  # type: ignore[attr-defined]
    panel.reset_cursor()
    for index in range(20):
        panel.apply_log({"stdout": [f"line {index}"], "stderr": []})
    panel.clear()

    table = _scrollable_table(host, (("a", "A", 80), ("b", "B", 120)), bus=app.bus)
    table.insert("", "end", values=("1", "2"))

    segmented = SegmentedControl(
        host,
        app.bus,  # type: ignore[attr-defined]
        (("one", "One"), ("two", "Two")),
        on_change=lambda _v: None,
        motion=app.motion,  # type: ignore[attr-defined]
    )
    segmented.set("two", animate=True)
    if segmented.get() != "two":
        raise AssertionError("SegmentedControl.set did not update the selection")
    segmented._step(1)  # type: ignore[attr-defined]

    dot = StatusDot(host, bus=app.bus)
    for state in ("ok", "warn", "error", "idle", "unknown-state"):
        dot.set_status(state)

    toasts = ToastHost(app, app.bus, app.motion)  # type: ignore[attr-defined]
    toasts.show("smoke toast", kind="ok")
    toasts.show("smoke toast 2", kind="warn", timeout_ms=50)

    animated_label = tk.Label(host, text="0.0")
    animated = AnimatedValue(animated_label, app.motion)  # type: ignore[attr-defined]
    animated.set("5.0", app.palette.ok, value=5.0, formatter=lambda value: f"{value:.1f}")  # type: ignore[attr-defined]
    animated.set("idle")

    ToolTip(row, "smoke tooltip", bus=app.bus)


def _exercise_page_handlers(app: object) -> None:
    """Call the selection handlers the tests cannot reach without a display."""

    empty = Event()
    app.show_page("Evaluations")
    app.pages["Evaluations"]._on_eval_select(empty)
    app.pages["Evaluations"]._on_checkpoint_select(empty)
    app.show_page("Runs / Checkpoints")
    app.pages["Runs / Checkpoints"]._on_select(empty)
    app.show_page("Settings")
    app.pages["Settings"]._on_select_ttk_mechanic(empty)
    app.pages["Settings"]._browse_godot()
    app.pages["Settings"]._change_output_root()
    app.show_page("Training")
    training = app.pages["Training"]
    training._on_agents(None, None)
    training._on_compatibility(None, None)
    training._on_log("", 0, None, None)
    app.show_page("Benchmarks")
    app.pages["Benchmarks"]._cancel()
    app.pages["Benchmarks"]._update_buttons()
    app.pages["Benchmarks"]._on_finished(None, RuntimeError("smoke"))
    app.show_page("Dashboard")
    app.pages["Dashboard"]._on_agents_summary(None, None)
    app.open_command_palette()
    app.set_status("smoke status", toast=True)
    app.notify("smoke notify", kind="info")
    try:
        app.show_page("No such page")
    except KeyError:
        pass
    else:
        raise AssertionError("show_page must reject an unknown page name")


def _callback_failures() -> list[str]:
    """Turn the recorded polling-loop exceptions into failure lines."""
    messages: list[str] = []
    for error, trace in zip(_CALLBACK_ERRORS, _CALLBACK_TRACES, strict=True):
        location = ""
        for line in trace.splitlines():
            if "control_center_pages.py" in line or "control_center_widgets.py" in line:
                location = line.strip()
        messages.append(f"background callback: {type(error).__name__}: {error}  [{location}]")
    return messages


def _report_failures(failures: list[str]) -> int:
    """Print the verdict; returns the process exit code."""
    if _CALLBACK_TRACES:
        print("--- background callback tracebacks ---")
        for trace in _CALLBACK_TRACES:
            print(trace)
    print()
    if failures:
        print(f"FAILURES ({len(failures)}): " + ", ".join(failures))
        return 1
    print("all smoke steps passed")
    return 0


def run_smoke() -> int:
    """Construct and drive the real Control Center against the fake Tk."""
    if str(PROJECT_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
    install_fake_tkinter()

    from sandboxai.adapter import SandboxAIAdapter
    from sandboxai.control_center_desktop import PAGE_CLASSES, PAGE_WIDGETS, ControlCenter

    root = pathlib.Path(tempfile.mkdtemp(prefix="cc-smoke-"))
    adapter = SandboxAIAdapter(project_root=root, output_root=root / "training")
    _capture_callback_errors()
    app = ControlCenter(adapter=adapter)
    print(f"construct: ok ({root})")

    failures: list[str] = []

    def sweep_pages() -> None:
        for page_class in PAGE_CLASSES:
            title = page_class.title
            app.show_page(title)
            app.pages[title].refresh()
            _drain(app)

    def sweep_styles() -> None:
        for name in ("cyan", "light", "corz"):
            app.set_theme(name)
        # A density change rebuilds a page around its widgets; what the
        # operator was reading has to survive that rebuild.
        app.show_page("Training")
        marker = "density marker 4711"
        app.pages["Training"].log_panel.apply_log({"stdout": [marker]})
        app.set_density("compact")
        surviving = app.pages["Training"].log_panel.text.get("1.0", "end-1c")
        if marker not in str(surviving):
            raise AssertionError("a density change wiped the log the operator was reading")
        for name in ("ultra", "comfort"):
            app.set_density(name)
        for name in ("reduced", "cinematic", "normal"):
            app.set_motion(name)

    def settings_and_pages() -> None:
        _exercise_shell(app, PAGE_CLASSES)
        _exercise_studio(app.pages["Settings"], PAGE_WIDGETS)
        _exercise_presets(app.pages["Settings"])
        _exercise_preset_transfer(app)
        _exercise_accent(app)
        _exercise_appearance(app.pages["Settings"])
        app.show_page("Training")
        _exercise_training(app.pages["Training"])
        _drain(app)
        app.show_page("Benchmarks")
        _exercise_benchmarks(app.pages["Benchmarks"])
        _drain(app)
        for title in ("Dashboard", "Evaluations", "Runs / Checkpoints", "System / Telemetry"):
            app.show_page(title)
            app.pages[title].refresh()
            _drain(app)

    _step(failures, "show and refresh every page", sweep_pages)
    _step(failures, "themes, densities, motion levels", sweep_styles)
    _step(failures, "settings, training, benchmark, telemetry", settings_and_pages)
    _step(failures, "reusable widgets", lambda: _exercise_widgets(app))
    _step(failures, "page handlers", lambda: _exercise_page_handlers(app))
    _step(failures, "close", app._on_close)

    failures.extend(_callback_failures())
    return _report_failures(failures)


if __name__ == "__main__":
    raise SystemExit(run_smoke())
