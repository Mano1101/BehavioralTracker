/* BehavioralTracker.exe -- runs the application INSIDE this process (no console window).

   The bundled Python (python\python312.dll) is loaded into BehavioralTracker.exe itself instead of
   starting python\pythonw.exe.  Windows therefore shows "BehavioralTracker" with the app's own icon
   on the taskbar, in Task Manager and in "Pin to taskbar" -- never "Python". */

#include <windows.h>
#include <wchar.h>

typedef int (*PyMainFn)(int, wchar_t **);

static void fail(const wchar_t *what, const wchar_t *path)
{
    wchar_t msg[8800];
    swprintf(msg, 8800, L"BehavioralTracker could not start:\n%ls\n%ls\n\nPlease reinstall BehavioralTracker.",
             what, path);
    MessageBoxW(NULL, msg, L"BehavioralTracker", MB_ICONERROR | MB_OK);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR args, int nShow)
{
    static const wchar_t *clear_vars[] = {
        L"PYTHONPATH", L"PYTHONSTARTUP", L"PYTHONUSERBASE", L"PYTHONEXECUTABLE", L"PYTHONPLATLIBDIR",
        L"PYTHONWARNINGS", L"PYTHONINSPECT", L"PYTHONOPTIMIZE", L"PYTHONDEBUG", L"PYTHONVERBOSE",
        L"PYTHONCASEOK", L"PYTHONSAFEPATH", L"PYTHONSTARTUP", L"PYTHONPYCACHEPREFIX", L"PYTHONSAFEPATH",
        L"VIRTUAL_ENV", L"CONDA_PREFIX", NULL };
    wchar_t dir[4096], pydir[4200], dll[4300], script[4300], appdir[4300];
    DWORD n = GetModuleFileNameW(NULL, dir, 4096);
    if (n == 0 || n >= 4096) return 1;
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    swprintf(pydir, 4200, L"%ls\\python", dir);
    swprintf(dll, 4300, L"%ls\\python312.dll", pydir);
    swprintf(script, 4300, L"%ls\\app\\main_pro.py", dir);
    swprintf(appdir, 4300, L"%ls\\app", dir);

    /* a private, isolated interpreter: our own home, no user site-packages, no outside PYTHON* settings */
    for (int i = 0; clear_vars[i]; i++) SetEnvironmentVariableW(clear_vars[i], NULL);
    SetEnvironmentVariableW(L"PYTHONHOME", pydir);
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetCurrentDirectoryW(appdir);
    SetDllDirectoryW(pydir);                       /* vcruntime140.dll etc. next to python312.dll */

    HMODULE py = LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!py) { fail(L"the program files are missing or damaged:", dll); return 1; }
    PyMainFn py_main = (PyMainFn)GetProcAddress(py, "Py_Main");
    if (!py_main) { fail(L"the program files are damaged:", dll); return 1; }

    /* argv = BehavioralTracker <script> [the arguments this exe was given] */
    int extra = 0;
    wchar_t **given = (args && *args) ? CommandLineToArgvW(args, &extra) : NULL;
    if (!given) extra = 0;
    wchar_t **argv = (wchar_t **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(wchar_t *) * (extra + 4));
    int argc = 0;
    argv[argc++] = L"BehavioralTracker";
    argv[argc++] = L"-s";
    argv[argc++] = script;
    for (int i = 0; i < extra; i++) argv[argc++] = given[i];
    argv[argc] = NULL;
    return py_main(argc, argv);
}
