"""Create Windows shortcuts that launch the GUI without a console window.

Puts "Sphere Diffraction.lnk" on the Desktop and in the Start Menu, targeting
the venv's pythonw.exe with the repo as working directory and
app/resources/icon.ico as the icon. Each shortcut also carries the same
AppUserModelID the app sets at runtime, so the taskbar button and pinned
icons resolve to the app icon instead of the Python launcher's.

Run from the repo root, inside the venv:

    python scripts/create_shortcut.py
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.__main__ import APP_USER_MODEL_ID

SHORTCUT_NAME = "Sphere Diffraction.lnk"
DESCRIPTION = "Mie scattering patterns and RCS for multilayer spheres"

_ole32 = ctypes.OleDLL("ole32")
_shell32 = ctypes.OleDLL("shell32")


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, guid_string: str):
        super().__init__()
        _ole32.CLSIDFromString(guid_string, ctypes.byref(self))


_ole32.CLSIDFromString.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", ctypes.c_ulong)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("pwszVal", ctypes.c_wchar_p),
        ("padding", ctypes.c_ubyte * 8),
    ]


VT_LPWSTR = 31
GPS_READWRITE = 2

PKEY_AppUserModel_ID = PROPERTYKEY(
    GUID("{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"), 5
)
IID_IPropertyStore = GUID("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")

FOLDERID_Desktop = GUID("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}")
FOLDERID_Programs = GUID("{A77F5D77-2E2B-44C3-A6A2-ABA601054A51}")


def _method(com_object: ctypes.c_void_p, index: int, *argtypes):
    vtable = ctypes.cast(
        com_object, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def known_folder(folder_id: GUID) -> Path:
    out = ctypes.c_wchar_p()
    _shell32.SHGetKnownFolderPath(
        ctypes.byref(folder_id), 0, None, ctypes.byref(out)
    )
    path = Path(out.value)
    _ole32.CoTaskMemFree(out)
    return path


def create_link(lnk: Path, target: Path, args: str, workdir: Path, icon: Path) -> None:
    script = f"""
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut('{lnk}')
$s.TargetPath = '{target}'
$s.Arguments = '{args}'
$s.WorkingDirectory = '{workdir}'
$s.IconLocation = '{icon},0'
$s.Description = '{DESCRIPTION}'
$s.Save()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )


def set_aumid(lnk: Path, aumid: str) -> None:
    """Stamp System.AppUserModel.ID onto the shortcut via IPropertyStore."""
    _ole32.CoInitialize(None)
    store = ctypes.c_void_p()
    _shell32.SHGetPropertyStoreFromParsingName(
        ctypes.c_wchar_p(str(lnk)),
        None,
        GPS_READWRITE,
        ctypes.byref(IID_IPropertyStore),
        ctypes.byref(store),
    )
    try:
        value = PROPVARIANT()
        value.vt = VT_LPWSTR
        buffer = ctypes.c_wchar_p(aumid)
        value.pwszVal = buffer
        set_value = _method(
            store, 6, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)
        )
        set_value(store, ctypes.byref(PKEY_AppUserModel_ID), ctypes.byref(value))
        commit = _method(store, 7)
        commit(store)
    finally:
        release = _method(store, 2)
        release(store)


def read_aumid(lnk: Path) -> str | None:
    _ole32.CoInitialize(None)
    store = ctypes.c_void_p()
    _shell32.SHGetPropertyStoreFromParsingName(
        ctypes.c_wchar_p(str(lnk)),
        None,
        0,
        ctypes.byref(IID_IPropertyStore),
        ctypes.byref(store),
    )
    try:
        value = PROPVARIANT()
        get_value = _method(
            store, 5, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)
        )
        get_value(store, ctypes.byref(PKEY_AppUserModel_ID), ctypes.byref(value))
        return value.pwszVal if value.vt == VT_LPWSTR else None
    finally:
        release = _method(store, 2)
        release(store)


def main() -> None:
    if sys.platform != "win32":
        sys.exit("Windows only.")

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        sys.exit(f"pythonw.exe not found next to {sys.executable}")

    icon = ROOT / "app" / "resources" / "icon.ico"
    if not icon.exists():
        sys.exit("app/resources/icon.ico missing — run scripts/generate_icon.py first")

    for folder in (known_folder(FOLDERID_Desktop), known_folder(FOLDERID_Programs)):
        lnk = folder / SHORTCUT_NAME
        create_link(lnk, pythonw, "-m app", ROOT, icon)
        set_aumid(lnk, APP_USER_MODEL_ID)
        stamped = read_aumid(lnk)
        status = "ok" if stamped == APP_USER_MODEL_ID else f"AUMID readback failed: {stamped!r}"
        print(f"created {lnk}  [{status}]")

    print(
        "\nLaunch the app from either shortcut (no console window). To pin it,\n"
        "right-click the shortcut and choose 'Pin to taskbar'. If Windows shows\n"
        "a stale icon, unpin and re-pin — the taskbar caches icons aggressively."
    )


if __name__ == "__main__":
    main()
