// Native Windows folder-picker bridge used by CASCADE Studio under WSL.
// The GUI compiles and loads this helper through PowerShell when available;
// users do not build or invoke it directly.
using System;
using System.Runtime.InteropServices;
public static class CascadePickerNative {
    [ComImport]
    [Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    private class FileOpenDialog { }

    [ComImport]
    [Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint sigdnName, out IntPtr name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem other, uint hint, out int order);
    }

    [ComImport]
    [Guid("42F85136-DB7E-439C-85F1-E4075D135FC8")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileDialog {
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint count, IntPtr filters);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem folder);
        void SetFolder(IShellItem folder);
        void GetFolder(out IShellItem folder);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, uint placement);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int error);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    private static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string path,
        IntPtr bindingContext,
        ref Guid interfaceId,
        out IShellItem item
    );

    private delegate bool EnumThreadDelegate(IntPtr hWnd, IntPtr lParam);
    [DllImport("kernel32.dll")] private static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] private static extern bool EnumThreadWindows(uint id, EnumThreadDelegate callback, IntPtr data);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
    public static void FocusDialog(IntPtr owner) {
        EnumThreadWindows(GetCurrentThreadId(), delegate(IntPtr hWnd, IntPtr data) {
            if (hWnd != owner && IsWindowVisible(hWnd)) { BringWindowToTop(hWnd); SetForegroundWindow(hWnd); }
            return true;
        }, IntPtr.Zero);
    }
    public static void CloseWindows(IntPtr owner) {
        EnumThreadWindows(GetCurrentThreadId(), delegate(IntPtr hWnd, IntPtr data) {
            if (hWnd != owner) PostMessage(hWnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
            return true;
        }, IntPtr.Zero);
        PostMessage(owner, 0x0010, IntPtr.Zero, IntPtr.Zero);
    }

    public static string PickFolder(IntPtr owner, string title, string initialFolder) {
        IFileDialog dialog = (IFileDialog)new FileOpenDialog();
        IShellItem initialItem = null;
        IShellItem resultItem = null;
        IntPtr displayName = IntPtr.Zero;
        try {
            uint options;
            dialog.GetOptions(out options);
            // PICKFOLDERS | FORCEFILESYSTEM | PATHMUSTEXIST | NOCHANGEDIR
            dialog.SetOptions(options | 0x20u | 0x40u | 0x800u | 0x8u);
            dialog.SetTitle(title);
            if (!String.IsNullOrWhiteSpace(initialFolder) && System.IO.Directory.Exists(initialFolder)) {
                Guid shellItemId = new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE");
                SHCreateItemFromParsingName(initialFolder, IntPtr.Zero, ref shellItemId, out initialItem);
                dialog.SetFolder(initialItem);
            }
            int status = dialog.Show(owner);
            if (status == unchecked((int)0x800704C7)) return String.Empty;
            Marshal.ThrowExceptionForHR(status);
            dialog.GetResult(out resultItem);
            resultItem.GetDisplayName(0x80058000u, out displayName);
            return Marshal.PtrToStringUni(displayName) ?? String.Empty;
        } finally {
            if (displayName != IntPtr.Zero) Marshal.FreeCoTaskMem(displayName);
            if (resultItem != null) Marshal.FinalReleaseComObject(resultItem);
            if (initialItem != null) Marshal.FinalReleaseComObject(initialItem);
            Marshal.FinalReleaseComObject(dialog);
        }
    }
}
