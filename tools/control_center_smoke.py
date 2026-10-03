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
        """Deliver an event the way Tk's bindtags do.

        Own bindings first, then the widget's class binding, then the
        containing toplevel. A handler that returns ``"break"`` (as the
        product code does) stops the propagation, which is exactly the
        contract a pre-bound widget relies on; returns the same string so a
        caller can assert on it.
        """
        # Tk sets %W to the widget the event was delivered to; handlers that
        # ask "am I inside this scroll area?" depend on it.
        kw = {"widget": self, **kw}
        if self._dispatch(sequence, kw) == "break":
            return "break"
        if self._tk_class_event(sequence) == "break":
            return "break"
        toplevel = self.winfo_toplevel()
        if toplevel is not self and toplevel._dispatch(sequence, kw) == "break":
            return "break"
        return None

    def _dispatch(self, sequence: str, kw: dict) -> object:
        for func in self._tk_binds.get(sequence, []):
            if func(Event(**kw)) == "break":
                return "break"
        return None

    def _tk_class_event(self, sequence: str) -> object:
        """Tk's class bindings; modelled only where the harness hid a bug."""
        return None

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
        """Destroy like Tk: children first, then this widget's own bindings.

        Real Tk fires ``<Destroy>`` for every widget it tears down, which is
        where cleanup lives (releasing scrollbar bindings, cancelling an
        animation, forgetting a toast). A stub that skipped the event made
        those handlers look unnecessary - the same blind spot that once hid
        two Tk-only bugs.
        """
        if getattr(self, "_tk_destroyed", False):
            return
        for child in list(self._tk_children):
            child.destroy()
        self._tk_children.clear()
        object.__setattr__(self, "_tk_destroyed", True)
        self.event_generate("<Destroy>")

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

    # Real widgets expose both of these; product code walks ``master`` to find
    # a theme bus or a scroll area's scope, so a ``_noop`` in their place would
    # silently stop every such walk at the first hop.
    @property
    def master(self):
        return self._tk_master

    def winfo_class(self):
        return type(self).__name__.lstrip("_")

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
    """The widget-level stacking calls of ``tkinter.Misc``.

    ``Canvas`` overrides ``lift``/``lower`` with the *item* operations, so
    product code reaches the widget-level call through ``tk.Misc`` (a canvas
    ``lower()`` with no tag is a Tcl error on a real build). The fake has to
    offer the same path, otherwise the smoke run fails where real Tk works.
    """

    @staticmethod
    def lift(widget, aboveThis=None):
        return _Base.lift(widget, aboveThis)

    @staticmethod
    def lower(widget, belowThis=None):
        return _Base.lower(widget, belowThis)


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

    def yview_scroll(self, number=0, what=None):
        # The scroll areas move the view with this call, so the harness has to
        # remember it: that is how a wheel check can see the page react.
        self._tk_options.setdefault("yview_scrolls", []).append((number, what))
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
        self._selected: set[int] = set()

    def insert(self, index, *elements):
        self._items.extend(elements)

    def delete(self, first, last=None):
        self._items.clear()

    def get(self, first, last=None):
        # Tk reads a single item with ``get(index)`` and a range with
        # ``get(first, "end")``; the palette does both.
        if last is None and isinstance(first, int):
            return self._items[first]
        return tuple(self._items)

    def size(self):
        return len(self._items)

    def curselection(self):
        return tuple(sorted(self._selected))

    def see(self, index):
        self._tk_options["see"] = index
        return None

    def activate(self, index):
        self._tk_options["active"] = index
        return None

    def selection_clear(self, first=0, last=None):
        self._selected.clear()
        return None

    def selection_set(self, first, last=None):
        self._selected.add(int(first))
        return None

    def index(self, spec):
        if spec == "end":
            return len(self._items)
        return int(spec)

    def _tk_class_event(self, sequence: str) -> object:
        """A real Listbox steps its selection on Up/Down (browse mode).

        Modelling this is what makes the harness catch arrows that are bound
        on the *toplevel* instead of on the list: the class binding runs
        first, the toplevel binding then steps a second time.
        """
        if sequence not in ("<Up>", "<Down>") or not self._items:
            return None
        step = 1 if sequence == "<Down>" else -1
        current = self.curselection()
        index = max(0, min(len(self._items) - 1, (current[0] if current else 0) + step))
        self.selection_clear(0, "end")
        self.selection_set(index)
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
        #: Every column write, so a check can prove a table is configured once
        #: and never re-fitted from the geometry it was just given.
        self._column_writes: list[tuple[tuple, dict]] = []

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
        if kw:
            self._column_writes.append((a, kw))
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
    """Type a custom hex accent, then fall back to the theme's own accent."""
    page = app.pages["Settings"]
    if "_accent_swatches" in vars(page):
        raise AssertionError("curated accent swatches should not be present")
    page.accent_var.set("#F5A524")
    page._apply_accent()  # type: ignore[attr-defined]
    if app.palette.accent.lower() != "#f5a524":
        raise AssertionError(f"a typed accent did not reach the palette: {app.palette.accent}")
    if app.prefs.accent.lower() != "#f5a524":
        raise AssertionError("a typed accent was not remembered in the preferences")
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
    """The UI exposes one automatic action; Cancel exists only during a run."""
    obsolete_controls = (
        "mode_control",
        "custom_env_var",
        "custom_worker_var",
        "custom_steps_var",
        "custom_minutes_var",
        "auto_train_var",
    )
    present = [name for name in obsolete_controls if name in vars(page)]
    if present:
        raise AssertionError(f"obsolete benchmark controls remain: {present}")
    if page.run_button.cget("text") != "Start Benchmark":  # type: ignore[attr-defined]
        raise AssertionError("the benchmark's only idle action must be Start Benchmark")
    if page.cancel_button.winfo_manager():  # type: ignore[attr-defined]
        raise AssertionError("Cancel must be hidden while the benchmark is idle")

    plan = page.current_plan()  # type: ignore[attr-defined]
    if plan["mode"] != "auto" or plan["errors"]:
        raise AssertionError(f"the benchmark must build a valid automatic plan: {plan}")
    if "Steps/s" not in page.LIVE_CARD_NAMES:  # type: ignore[attr-defined]
        raise AssertionError("live telemetry must include Steps/s")
    if "peak fps" in page.LIVE_CARD_NAMES:  # type: ignore[attr-defined]
        raise AssertionError("duplicate Peak FPS telemetry should be removed")

    from threading import Event

    page._running = True  # type: ignore[attr-defined]
    page._cancel_event = Event()  # type: ignore[attr-defined]
    page._update_buttons()  # type: ignore[attr-defined]
    if page.cancel_button.winfo_manager() != "pack":  # type: ignore[attr-defined]
        raise AssertionError("Cancel must appear while a benchmark is running")
    if str(page.cancel_button.cget("state")) == "disabled":  # type: ignore[attr-defined]
        raise AssertionError("Cancel must be enabled before cancellation is requested")
    page._cancel()  # type: ignore[attr-defined]
    if not page._cancel_event.is_set():  # type: ignore[attr-defined]
        raise AssertionError("Cancel did not reach the running benchmark")
    if str(page.cancel_button.cget("state")) != "disabled":  # type: ignore[attr-defined]
        raise AssertionError("Cancel must disable after it has been requested")
    page._running = False  # type: ignore[attr-defined]
    page._cancel_event = None  # type: ignore[attr-defined]
    page._update_buttons()  # type: ignore[attr-defined]
    if page.cancel_button.winfo_manager():  # type: ignore[attr-defined]
        raise AssertionError("Cancel must hide when the benchmark finishes")



def _exercise_board_resize(app: object, host: object) -> None:
    """The card board's resize decision must be debounced and hysteretic.

    The desktop suite hung inside ``update()`` because a board re-decided its
    column count from the ``<Configure>`` event of the layout that decision
    had just produced. This drives the same handler the fake Tk never fires:
    a width inside the hysteresis band must change nothing, a width that
    clears it must schedule exactly one application, and a repeated width must
    not schedule anything at all.
    """
    import tkinter as tk

    from sandboxai.control_center_layout import WidgetSpec, default_layout
    from sandboxai.control_center_ui import LayoutBoard, LayoutBus

    specs = (
        WidgetSpec("one", "One"),
        WidgetSpec("two", "Two"),
        WidgetSpec("three", "Three"),
        WidgetSpec("four", "Four"),
    )
    board = LayoutBoard(
        host,  # type: ignore[arg-type]
        LayoutBus(default_layout({"Scratch": specs})),
        page_key="Scratch",
        specs=specs,
        columns=3,
        gap=4,
        min_column_width=320,
    )
    board.pack(fill="both")
    for spec in specs:
        board.add(spec.widget_id, lambda parent, _spec=spec: tk.Frame(parent))
    board.rebuild()
    if board.columns() != 3:
        raise AssertionError("a fresh board did not start with the page's columns")

    # A width inside the shrink band (it wants 2 columns, but not by enough)
    # must leave the current count and the pending count alone.
    board._on_configure(Event(width=940))  # type: ignore[attr-defined]
    if board._pending_columns is not None:  # type: ignore[attr-defined]
        raise AssertionError("a width inside the hysteresis band scheduled a re-grid")

    # A width that clears the band schedules the new count, but must not apply
    # it before the debounce elapses.
    board._on_configure(Event(width=700))  # type: ignore[attr-defined]
    if board._pending_columns != 2:  # type: ignore[attr-defined]
        raise AssertionError("a width that clears the band did not schedule a re-grid")
    if board.columns() != 3:
        raise AssertionError("the board re-gridded before the debounce elapsed")
    if board._resize_job is None:  # type: ignore[attr-defined]
        raise AssertionError("the board scheduled a re-grid without a debounce timer")

    board._apply_columns()  # type: ignore[attr-defined]
    if board.columns() != 2:
        raise AssertionError("the board kept its column count on a narrower width")

    # Same width, or a width that leads to the same count: nothing scheduled.
    board._on_configure(Event(width=760))  # type: ignore[attr-defined]
    if board._pending_columns is not None or board._resize_job is not None:  # type: ignore[attr-defined]
        raise AssertionError("a width leading to the current count scheduled a re-grid")

    # An unchanged rebuild must not touch a single widget.
    slots = board._slots  # type: ignore[attr-defined]
    board.rebuild()
    if board._slots != slots:  # type: ignore[attr-defined]
        raise AssertionError("an unchanged rebuild changed the board's placement")
    if board._last_width != 760:  # type: ignore[attr-defined]
        raise AssertionError("the board did not remember the last width it saw")

    board.destroy()


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

    table = _scrollable_table(
        host,
        (("a", "A", 80), ("b", "B", 120)),
        bus=app.bus,
        empty_text="Nothing here yet.",
    )
    # The empty state is an overlay, not a row: it appears while the table has
    # no rows and disappears the moment one arrives.
    table.empty_state.refresh()  # type: ignore[attr-defined]
    if not table.empty_state.is_shown():  # type: ignore[attr-defined]
        raise AssertionError("an empty table did not show its empty-state hint")
    table.insert("", "end", values=("1", "2"))
    table.empty_state.refresh()  # type: ignore[attr-defined]
    if table.empty_state.is_shown():  # type: ignore[attr-defined]
        raise AssertionError("the empty-state hint stayed over a filled table")
    table.empty_state.set_text("Changed hint")  # type: ignore[attr-defined]
    if table.empty_state.text != "Changed hint":  # type: ignore[attr-defined]
        raise AssertionError("the empty-state hint text was not updated")

    _exercise_board_resize(app, host)

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

    _exercise_scroll_area(app, host)


def _exercise_scroll_area(app: object, host: object) -> None:
    """The wheel must scroll a page from anywhere over its content.

    Tk delivers the wheel to the widget under the pointer, so the page used
    to scroll only when the pointer happened to be over the canvas itself;
    over a card's labels nothing moved and the scrollbar looked like the only
    way down. The fake now delivers like Tk's bindtags (own bindings, class,
    toplevel), which is what makes this check meaningful.
    """
    import tkinter as tk

    from sandboxai.control_center_ui import ScrollArea

    area = ScrollArea(host, app.bus, scale_px=app.px)  # type: ignore[attr-defined]
    area.pack()
    label = tk.Label(area.body, text="card label")
    label.pack()

    def scrolls() -> list:
        return list(area.canvas._tk_options.get("yview_scrolls", []))

    label.event_generate("<MouseWheel>", delta=-120)
    if not scrolls():
        raise AssertionError("the wheel over card content must scroll the page")
    # A widget that scrolls itself keeps the wheel; the page must stay put.
    text = tk.Text(area.body)
    text.pack()
    before = len(scrolls())
    text.event_generate("<MouseWheel>", delta=-120)
    if len(scrolls()) != before:
        raise AssertionError("the wheel over a Text inside a page must not scroll the page")
    # And a wheel outside the area must leave it alone.
    outside = tk.Label(host, text="outside")
    outside.pack()
    before = len(scrolls())
    outside.event_generate("<MouseWheel>", delta=-120)
    if len(scrolls()) != before:
        raise AssertionError("a wheel outside the page must not scroll it")


def _assert_theme_listeners_do_not_leak(app: object) -> None:
    """Rebuilt pages must not leave dead theme listeners behind.

    A density change destroys and recreates a page's widgets. Each of them
    subscribes to the theme bus, so without ownership the listener list grew
    by ~190 per change (97 -> 667 after three) and every later theme change
    walked the corpses. The bus now drops listeners whose owner is gone; the
    list may hold the dead ones until the next notification, but it must not
    keep growing.
    """
    app.set_theme(app.prefs.theme)
    settled = len(app.bus._listeners)
    for _ in range(3):
        app.set_density("compact")
        app.set_density("comfort")
    app.set_theme(app.prefs.theme)
    pruned = len(app.bus._listeners)
    if pruned > settled:
        raise AssertionError(
            f"theme listeners leak across rebuilds: {settled} live -> {pruned} after "
            "three density sweeps"
        )


def _assert_motion_survives_a_dead_widget(app: object) -> None:
    """One destroyed widget must not stop every animation in the window.

    A density change rebuilds a page while its animations (a segmented
    control's indicator, an animated number) may still be mid-flight. Their
    next frame then paints a destroyed widget and Tk raises ``TclError`` -
    inside the shared 16 ms ticker, which used to stop scheduling itself,
    silently freezing every remaining animation.
    """
    import tkinter as tk

    frames: list[float] = []

    def exploding(_progress: float) -> None:
        raise tk.TclError("application has been destroyed")

    app.motion.tween(40, exploding)
    app.motion.tween(40, frames.append)
    app.motion._tick()  # must not raise: the ticker isolates each callback

    if not frames:
        raise AssertionError("a raising tween stopped the healthy one from running")
    if not app.motion.running and any(
        tween.on_frame is frames.append for tween in app.motion._tweens.values()
    ):
        raise AssertionError("the ticker stopped scheduling while a tween was still live")
    app.motion.stop_all()


def _assert_animations_release_the_ticker(app: object) -> None:
    """A destroyed widget must hand its animation back.

    A pulsing status dot keeps the shared 16 ms ticker alive. If destroying
    the widget left the loop registered, a rebuilt page would keep the whole
    window awake at 60 fps for nothing, and the loop would paint a dead
    canvas on every frame.
    """
    import tkinter as tk

    from sandboxai.control_center_ui import MotionController, StatusDot

    host = tk.Frame(app)
    motor = MotionController(app, "normal")
    dot = StatusDot(host, app.bus, size=8)
    dot.set_state(app.bus.theme.ok, pulse=True, motion=motor)
    if not motor.running or not motor._loops:
        raise AssertionError("a pulsing status dot must start the shared ticker")

    host.destroy()

    if motor._loops or motor._tweens:
        raise AssertionError("a destroyed widget left its animation registered")
    # The tick that was already scheduled still runs (Tk cannot un-schedule
    # it retroactively); what matters is that it is the last one.
    motor._tick()
    if motor.running:
        raise AssertionError("the ticker kept scheduling after its last animation went away")
    motor.stop_all()


def _assert_every_widget_uses_the_app_bus(app: object) -> None:
    """No widget may be left on the module's default palette.

    A widget constructed without a bus silently subscribes to the *default*
    theme, so it keeps Corz colours after the operator switches to Light or
    changes the accent - invisible in a headless check and easy to miss in a
    screenshot. Every widget that carries a bus must therefore carry the
    application's own.
    """

    from sandboxai.control_center_ui import ThemeBus

    app_bus = app.bus
    offenders: list[str] = []
    roots = [app, *app.pages.values()]
    for root in roots:
        pending = list(root.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            for attribute in ("_bus", "bus"):
                value = getattr(widget, attribute, None)
                if isinstance(value, ThemeBus) and value is not app_bus:
                    offenders.append(
                        f"{widget.winfo_class()} has a foreign theme bus on .{attribute}"
                    )
    if offenders:
        raise AssertionError("; ".join(sorted(set(offenders))))


def _assert_every_page_attaches_its_cards(app: object) -> None:
    """No page may lay its cards out inside an unattached board.

    The harness runs without a display, so it cannot ask Tk whether a widget
    is *mapped*. It can ask which geometry manager owns it, and that is the
    exact regression: a board that ``Page.board()`` builds and fills but
    never packs, or cards the board never grids, render as a page with a
    heading and nothing under it.
    """

    problems: list[str] = []
    for title, page in app.pages.items():  # type: ignore[attr-defined]
        board = getattr(page, "_board", None)
        if board is None:
            continue
        if not board.winfo_manager():
            problems.append(f"{title}: the layout board has no geometry manager")
            continue
        attached = [child for child in board.winfo_children() if child.winfo_manager() == "grid"]
        if not attached:
            problems.append(f"{title}: no card on the board is attached")
    if problems:
        raise AssertionError("; ".join(problems))


def _assert_only_the_visible_page_polls(app: object) -> None:
    """One poll tick refreshes the visible page and nothing else.

    Every ``Page.refresh()`` submits background reads - run directories,
    benchmark history, the replay list. Running all eight on every 600 ms tick
    would keep reading artifacts for a window the operator is not looking at,
    which is the difference between a GUI that idles and one that keeps a
    laptop fan busy. The harness counts the calls instead of timing them, so
    the check is deterministic on any machine.
    """

    counters: dict[str, int] = {}
    originals: dict[str, object] = {}
    for title, page in app.pages.items():  # type: ignore[attr-defined]
        counters[title] = 0
        originals[title] = page.refresh

        def counting(_title: str = title, _original: object = originals[title]) -> None:
            counters[_title] += 1
            _original()  # type: ignore[operator]

        page.refresh = counting  # type: ignore[method-assign]
    try:
        app.show_page("System / Telemetry")  # type: ignore[attr-defined]
        for title in counters:
            counters[title] = 0
        app._tick()  # type: ignore[attr-defined]
        if counters["System / Telemetry"] != 1:
            raise AssertionError(
                "the visible page must refresh exactly once per poll tick, "
                f"got {counters['System / Telemetry']}"
            )
        offscreen = sorted(
            title for title, count in counters.items() if title != "System / Telemetry" and count
        )
        if offscreen:
            raise AssertionError(f"the poll tick refreshed hidden pages: {offscreen}")
    finally:
        for title, page in app.pages.items():  # type: ignore[attr-defined]
            page.refresh = originals[title]  # type: ignore[method-assign]


def _exercise_command_palette(app: object, window: object) -> None:
    """Drive the palette keys on the fake widgets.

    The fake ``event_generate`` only walks the bindings of the widget it is
    called on, so this covers the entry bindings (Down/Up/Return), the
    "highlight survives a key release" rule, and the list bindings that keep
    the arrows working once the operator clicked into the list.
    """

    def child(kind: str) -> object:
        for candidate in window.winfo_children():  # type: ignore[attr-defined]
            if type(candidate).__name__.lstrip("_").endswith(kind):
                return candidate
        raise AssertionError(f"the command palette has no {kind}")

    entry = child("Entry")
    listing = child("Listbox")
    if listing.curselection() != (0,):
        raise AssertionError("the palette must open with the first page highlighted")
    entry.event_generate("<Down>")
    if listing.curselection() != (1,):
        raise AssertionError("Down in the palette entry must move the highlight")
    # Every key release runs the filter refresh; it must not reset the row.
    entry.event_generate("<KeyRelease>")
    if listing.curselection() != (1,):
        raise AssertionError("a key release must not undo the arrow-key move")
    listing.event_generate("<Up>")
    if listing.curselection() != (0,):
        raise AssertionError("Up on the palette list must move the highlight back")
    listing.event_generate("<Down>")
    if listing.curselection() != (1,):
        raise AssertionError("the list binding must suppress Tk's own cursor step")
    if listing.get(listing.curselection()[0]) != "Training":
        raise AssertionError("the highlighted row must be the second page")
    entry.event_generate("<Return>")
    if window.winfo_exists():
        raise AssertionError("Return must close the palette after choosing a page")
    if app._current is None or app._current.title != "Training":
        raise AssertionError("Return must switch to the highlighted page")


def _exercise_palette_shortcuts(app: object) -> None:
    """The palette's own Escape, Ctrl+K and Ctrl+1..7 bindings."""

    def reopen() -> object:
        app.open_command_palette()
        fresh = app._palette_window
        if fresh is None or not fresh.winfo_exists():
            raise AssertionError("Ctrl+K must reopen the palette after it closed")
        return fresh

    # Escape and Ctrl+K close it again; Ctrl+1..7 jump straight to a page.
    reopen().event_generate("<Escape>")
    if app._palette_window.winfo_exists():
        raise AssertionError("Escape must close the palette")
    reopen().event_generate("<Control-Key-k>")
    if app._palette_window.winfo_exists():
        raise AssertionError("Ctrl+K must close the palette from inside it")
    reopen().event_generate("<Control-Key-4>")
    if app._palette_window.winfo_exists():
        raise AssertionError("a page accelerator must close the palette")
    if app._current is None or app._current.title != "Evaluations":
        raise AssertionError("Ctrl+4 must switch to the fourth page")
    app.show_page("Dashboard")


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
    _exercise_stats(app)
    app.open_command_palette()
    first = app._palette_window
    if first is None or not first.winfo_exists():
        raise AssertionError("Ctrl+K must open a command palette window")
    app.open_command_palette()
    if app._palette_window is not first:
        raise AssertionError("a second Ctrl+K must reuse the open palette, not stack one")
    _exercise_command_palette(app, first)
    _exercise_palette_shortcuts(app)
    app.set_status("smoke status", toast=True)
    app.notify("smoke notify", kind="info")
    try:
        app.show_page("No such page")
    except KeyError:
        pass
    else:
        raise AssertionError("show_page must reject an unknown page name")


def _count_table_writes(tree: object) -> dict[str, int]:
    """Wrap a Treeview's delete/insert so a check can see rebuilds."""
    counts = {"delete": 0, "insert": 0}
    original_delete = tree.delete  # type: ignore[attr-defined]
    original_insert = tree.insert  # type: ignore[attr-defined]

    def delete(*items):  # type: ignore[no-untyped-def]
        counts["delete"] += 1
        return original_delete(*items)

    def insert(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["insert"] += 1
        return original_insert(*args, **kwargs)

    tree.delete = delete  # type: ignore[method-assign]
    tree.insert = insert  # type: ignore[method-assign]
    return counts


def _all_tables(widget: object) -> list:
    """Every Treeview in the window, however deep."""
    found = [widget] if isinstance(widget, _Treeview) else []
    for child in widget.winfo_children():  # type: ignore[attr-defined]
        found.extend(_all_tables(child))
    return found


def _assert_tables_do_not_rewrite_their_columns(app: object) -> None:
    """A table's columns are written once, when the table is built.

    The desktop suite hung on the benchmark results table: a ``<Configure>``
    handler re-fitted its columns to the width it had just been given, which
    changed the width the table asked for and produced the next Configure -
    396 489 of them inside the test's window. Tk's own ``stretch`` does the
    fill-the-spare-width job inside the widget, so nothing in Python may write
    a column width after the table exists. Showing and refreshing every page
    must write none.
    """

    from sandboxai.control_center_desktop import PAGE_CLASSES

    tables = _all_tables(app)
    if not tables:
        raise AssertionError("no table was found to check")
    before = {id(table): len(table._column_writes) for table in tables}
    for page_class in PAGE_CLASSES:
        app.show_page(page_class.title)  # type: ignore[attr-defined]
        app.pages[page_class.title].refresh()  # type: ignore[attr-defined]
        _drain(app)  # type: ignore[arg-type]
    offenders = [
        f"{type(table).__name__}({table.cget('columns')})"  # type: ignore[attr-defined]
        for table in tables
        if len(table._column_writes) > before[id(table)]
    ]
    if offenders:
        raise AssertionError(
            "a page re-fitted table columns after the table was built: " + ", ".join(offenders)
        )


def _assert_unchanged_tables_are_not_rebuilt(app: object) -> None:
    """A poll result identical to the last one must not rebuild the table.

    Rebuilding costs a delete plus one insert per row, and in real Tk it also
    drops the row the operator had selected. Every table that is fed from a
    poll now compares a signature of the rendered values and leaves an
    unchanged table alone; this drives the two inventories (and the runs
    table) with the same result twice and then with a real change.
    """

    evaluations = app.pages["Evaluations"]  # type: ignore[attr-defined]
    checkpoints = [
        {
            "run_id": "run-a",
            "kind": "latest",
            "path": "/tmp/run-a/checkpoints/latest.zip",
            "bytes": 1024,
            "modified_utc": "2026-10-01T00:00:00Z",
        }
    ]
    summaries = [
        {
            "path": "/tmp/run-a/evaluations/latest.json",
            "timesteps": 1000,
            "episodes": 10,
            "mean_episode_reward": 1.0,
            "win_rate": 0.5,
            "loss_rate": 0.25,
            "modified_utc": "2026-10-01T00:00:00Z",
        }
    ]
    evaluations._on_checkpoints(checkpoints, None)
    evaluations._on_evaluations(summaries, None)
    checkpoint_writes = _count_table_writes(evaluations.checkpoint_tree)
    evaluation_writes = _count_table_writes(evaluations.eval_tree)

    evaluations._on_checkpoints([dict(entry) for entry in checkpoints], None)
    evaluations._on_evaluations([dict(entry) for entry in summaries], None)
    if checkpoint_writes["delete"] or checkpoint_writes["insert"]:
        raise AssertionError("an unchanged checkpoint table was rebuilt")
    if evaluation_writes["delete"] or evaluation_writes["insert"]:
        raise AssertionError("an unchanged evaluation table was rebuilt")

    evaluations._on_evaluations([dict(summaries[0], mean_episode_reward=2.0)], None)
    if not evaluation_writes["insert"]:
        raise AssertionError("a changed evaluation must rebuild its table")
    if evaluation_writes["delete"] != 1:
        raise AssertionError("a rebuild must clear the table exactly once")

    runs = app.pages["Runs / Checkpoints"]  # type: ignore[attr-defined]
    report = {
        "run_id": "run-a",
        "status": {"state": "running"},
        "progress": {"fraction": 0.5},
        "checkpoints": {"count": 1},
        "config": {"device": "cpu", "environment_count": 8, "env_workers": 1},
        "evaluation": {"latest": {"mean_episode_reward": 1.0, "win_rate": 0.5}},
        "modified_utc": "2026-10-01T00:00:00Z",
        "run_dir": "/tmp/run-a",
    }
    runs._on_runs({"runs": [report], "run_count": 1}, None)
    run_writes = _count_table_writes(runs.tree)
    runs._on_runs({"runs": [dict(report)], "run_count": 1}, None)
    if run_writes["delete"] or run_writes["insert"]:
        raise AssertionError("an unchanged run table was rebuilt")
    runs._on_runs({"runs": [dict(report, status={"state": "finished"})], "run_count": 1}, None)
    if not run_writes["insert"]:
        raise AssertionError("a changed run state must rebuild the runs table")

    # Registering the same Treeview tag on every rebuild (the Evaluations
    # table does exactly that) must not grow the theme bookkeeping: the list
    # is replayed entry by entry on every theme switch, so it is a leak and a
    # slowdown in one.
    before = len(evaluations._tag_roles)
    for reward in (0.1, 0.2, 0.3, 0.4, 0.5):
        evaluations._on_evaluations([dict(summaries[0], mean_episode_reward=reward)], None)
    if len(evaluations._tag_roles) != before:
        raise AssertionError(
            f"re-registering a tag grew the theme list ({before} -> {len(evaluations._tag_roles)})"
        )


def _exercise_stats(app: object) -> None:
    """Drive the Stats page with a synthetic detailed replay.

    The page is the one place that decodes a recording end to end (header ->
    tick -> the 106-float vector -> contacts/objects/hearing/action), so the
    smoke run feeds it a replay shaped exactly like ``adapter.replay_stats``
    returns instead of only checking that the page exists.
    """

    app.show_page("Stats")  # type: ignore[attr-defined]
    stats = app.pages["Stats"]  # type: ignore[attr-defined]
    stats._on_replays([], None)
    stats._on_replay_stats(None, RuntimeError("smoke"))
    stats._on_evidence(None, None)
    stats._step_tick(1)
    stats._jump_to_tick()
    stats._on_replay_selected(Event())

    header = {
        "seed": 7,
        "map_id": "blind_corner",
        "scenario": "corner_fight",
        "lighting": "low_light",
        "curriculum_level": 6,
        "enemy_count": 2,
        "detail": "detailed",
        "policy_id": "smoke-brain",
    }
    replay = {
        "path": "/tmp/smoke/replays/episode_0001.jsonl",
        "name": "episode_0001.jsonl",
        "run": "run-smoke",
        "header": header,
        "tick_count": 3,
        "detailed": True,
        "tick_index": 1,
        "action": [2, 1, 0, 1, 1, 0],
        "reward": 0.25,
        "done": False,
        "observation": [0.5] * 106,
        "events": [{"kind": "combat", "tick": 1, "data": {"damage_taken": 5.0}}],
    }
    stats._on_replay_stats(replay, None)
    if "health" not in str(stats.summary_label.cget("text")):
        raise AssertionError("the Stats page must decode an observation into a summary")
    if not stats.vector_tree.get_children():
        raise AssertionError("the Stats page must render the observation vector table")
    if len(stats.vector_tree.get_children()) < 5:
        raise AssertionError("the vector table must group the fields into sections")
    stats._on_replays(
        [
            {
                "path": replay["path"],
                "name": replay["name"],
                "run": replay["run"],
                "header": header,
                "ticks": 3,
            }
        ],
        None,
    )
    stats._on_evidence(
        {
            "target": "Roblox TTK Testing",
            "evidence_checked_on": "2026-10-01",
            "mechanics": {
                "verified": [
                    {
                        "mechanic": "fire",
                        "status": "verified",
                        "implementation_rule": "keep",
                        "source_label": "official",
                    }
                ],
                "calibration_required": [],
                "excluded": [],
            },
        },
        None,
    )
    if "verified" not in str(stats.evidence_label.cget("text")):
        raise AssertionError("the Stats page must show the TTK evidence counts")

    # The rescan timer must keep running while a replay is selected: a
    # recording written during the session has to appear without the operator
    # pressing Rescan. The old code only scanned while no replay was loaded,
    # so a list that was already populated could never grow. Counting the
    # submissions (not the thread results) keeps the check deterministic.
    submissions = {"replays": 0, "ttk-evidence": 0}
    original_submit_poll = stats.submit_poll

    def counting_submit_poll(operation, fn, callback):  # type: ignore[no-untyped-def]
        if operation in submissions:
            submissions[operation] += 1
        original_submit_poll(operation, fn, callback)

    stats.submit_poll = counting_submit_poll  # type: ignore[method-assign]
    stats._poll_count = 0
    stats.refresh()
    if submissions["replays"] != 1:
        raise AssertionError("a tick with a selected replay must still rescan the folder")
    stats._poll_count = stats.RESCAN_EVERY
    stats.refresh()
    if submissions["replays"] != 2:
        raise AssertionError("the rescan timer must keep running on later ticks")
    if submissions["ttk-evidence"] != 0:
        raise AssertionError("the evidence manifest must not be re-polled once it has loaded")

    # A scan that finds nothing must clear the decoded replay instead of
    # leaving its values on screen as if they were still backed by a file.
    stats._on_replays([], None)
    if stats._replay is not None:
        raise AssertionError("an empty rescan must clear the decoded replay")
    if "health" in str(stats.summary_label.cget("text")):
        raise AssertionError("an empty rescan must clear the decoded summary")
    if not stats._evidence_loaded:
        raise AssertionError("the evidence card must stop polling once it has a result")


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
        for page_class in PAGE_CLASSES:
            app.show_page(page_class.title)
            _assert_every_widget_uses_the_app_bus(app)

    _step(failures, "show and refresh every page", sweep_pages)
    _step(
        failures,
        "every page attaches its cards",
        lambda: _assert_every_page_attaches_its_cards(app),
    )
    _step(failures, "themes, densities, motion levels", sweep_styles)
    _step(failures, "settings, training, benchmark, telemetry", settings_and_pages)
    _step(
        failures,
        "theme listeners are pruned with their widgets",
        lambda: _assert_theme_listeners_do_not_leak(app),
    )
    _step(
        failures,
        "only the visible page polls",
        lambda: _assert_only_the_visible_page_polls(app),
    )
    _step(
        failures,
        "one dead widget cannot stop the ticker",
        lambda: _assert_motion_survives_a_dead_widget(app),
    )
    _step(
        failures,
        "destroyed widgets release their animations",
        lambda: _assert_animations_release_the_ticker(app),
    )
    _step(failures, "reusable widgets", lambda: _exercise_widgets(app))
    _step(
        failures,
        "an unchanged table is not rebuilt",
        lambda: _assert_unchanged_tables_are_not_rebuilt(app),
    )
    _step(
        failures,
        "tables do not re-fit their columns",
        lambda: _assert_tables_do_not_rewrite_their_columns(app),
    )
    _step(failures, "page handlers", lambda: _exercise_page_handlers(app))
    _step(failures, "close", app._on_close)

    failures.extend(_callback_failures())
    return _report_failures(failures)


if __name__ == "__main__":
    raise SystemExit(run_smoke())
