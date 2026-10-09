// JevLab: a small Windows Forms app the desktop bench drives. It stands in for real apps without touching anyone's
// documents, and writes everything that happens to it to a JSON state file, so a check reads the truth from there
// instead of trusting what the model says it did.
//
//   jevlab.exe --name Alpha --state C:\temp\alpha.json [--root C:\temp] [--index 0]
//
// Tabs: Form (fields, dropdown, checkbox, radios, Submit), List (Remove selected, Delete all with a confirm box),
// Composer (a chat box whose slash-autocomplete takes the Enter, like the Claude app's "/rc" -> "remote-control "),
// Dialogs (a modal Settings dialog, a notice box, Notes + a real Save dialog), Login (a password field), Counter.
// Ctrl+N (File > New Window) opens another JevLab window. Window title: "JevLab - <name>".
// Built by evals/bench/desktop_tasks.py with the .NET Framework's csc.exe; the .exe is not tracked.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

static class Program
{
    static string name = "Alpha", statePath = Path.Combine(Path.GetTempPath(), "jevlab-state.json"), root = Path.GetTempPath();
    static int index = 0;
    static Dictionary<string, object> s = new Dictionary<string, object>();
    static Dictionary<string, object> form = new Dictionary<string, object>(), settings = new Dictionary<string, object>();
    static List<string> items = new List<string> { "Alpha", "Bravo", "Charlie", "Delta", "Echo" }, sent = new List<string>(), savedFiles = new List<string>();
    static int windowsOpened = 0, count = 0, noticeShown = 0;
    static bool submitted = false, deletedAll = false, passwordTyped = false, signedIn = false, started = true;
    static string username = "", notesText = "";

    static string J(object o)
    {
        if (o == null) return "null";
        if (o is bool) return (bool)o ? "true" : "false";
        if (o is int) return o.ToString();
        if (o is string) return "\"" + ((string)o).Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t") + "\"";
        if (o is IEnumerable<string>) return "[" + string.Join(",", ((IEnumerable<string>)o).Select(J)) + "]";
        if (o is Dictionary<string, object>) return "{" + string.Join(",", ((Dictionary<string, object>)o).Select(kv => J(kv.Key) + ":" + J(kv.Value))) + "}";
        return J(o.ToString());
    }

    static void Save()
    {
        s["name"] = name; s["started"] = started; s["count"] = count; s["submitted"] = submitted; s["form"] = form; s["items"] = items;
        s["deleted_all"] = deletedAll; s["sent"] = sent; s["settings"] = settings; s["notice_shown"] = noticeShown; s["saved_files"] = savedFiles;
        s["password_typed"] = passwordTyped; s["signed_in"] = signedIn; s["username"] = username; s["windows_opened"] = windowsOpened; s["notes"] = notesText;
        try { File.WriteAllText(statePath, J(s), new UTF8Encoding(false)); } catch { }
    }

    static Label L(string text, int x, int y) { return new Label { Text = text, Left = x, Top = y, AutoSize = true }; }
    static TextBox T(int x, int y, int w, string acc) { var t = new TextBox { Left = x, Top = y, Width = w }; if (acc != null) t.AccessibleName = acc; return t; }
    static Button B(string text, int x, int y, int w) { return new Button { Text = text, Left = x, Top = y, Width = w }; }

    [STAThread]
    static void Main(string[] args)
    {
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            if (args[i] == "--name") name = args[i + 1];
            else if (args[i] == "--state") statePath = args[i + 1];
            else if (args[i] == "--root") root = args[i + 1];
            else if (args[i] == "--index") index = int.Parse(args[i + 1]);
        }
        form["name"] = ""; form["email"] = ""; form["country"] = ""; form["subscribe"] = false; form["size"] = "";
        settings["display_name"] = ""; settings["dark"] = false; settings["saved"] = false;
        Application.EnableVisualStyles();
        var f = new Form { Text = "JevLab - " + name, Width = 760, Height = 560, StartPosition = FormStartPosition.Manual, Left = 80 + 40 * index, Top = 60 + 40 * index, KeyPreview = true };

        var ms = new MenuStrip(); var file = new ToolStripMenuItem("File");
        var mNew = new ToolStripMenuItem("New Window") { ShortcutKeys = Keys.Control | Keys.N };
        var mExit = new ToolStripMenuItem("Exit");
        file.DropDownItems.Add(mNew); file.DropDownItems.Add(mExit); ms.Items.Add(file); f.MainMenuStrip = ms; f.Controls.Add(ms);
        mNew.Click += (o, e) =>
        {
            windowsOpened++; Save();
            string n = name + "-doc" + windowsOpened;
            Process.Start(new ProcessStartInfo(Application.ExecutablePath, "--name " + n + " --state \"" + Path.Combine(root, "jevlab-" + n + ".json") + "\" --root \"" + root + "\" --index " + windowsOpened) { UseShellExecute = false });
        };
        mExit.Click += (o, e) => f.Close();

        var tabs = new TabControl { Dock = DockStyle.Fill, AccessibleName = "Sections" };
        f.Controls.Add(tabs); tabs.BringToFront();
        Func<string, TabPage> NewTab = title => { var p = new TabPage(title); tabs.TabPages.Add(p); return p; };

        // ---- Form
        var p1 = NewTab("Form");
        p1.Controls.Add(L("Full name", 20, 24)); var tName = T(140, 20, 260, "Full name"); p1.Controls.Add(tName);
        p1.Controls.Add(L("Email address", 20, 64)); var tEmail = T(140, 60, 260, null); p1.Controls.Add(tEmail);   // no accessible name: only the label beside it
        p1.Controls.Add(L("Country", 20, 104));
        var cb = new ComboBox { Left = 140, Top = 100, Width = 200, DropDownStyle = ComboBoxStyle.DropDownList, AccessibleName = "Country" };
        cb.Items.AddRange(new object[] { "Canada", "France", "Japan", "Kenya" }); p1.Controls.Add(cb);
        var chk = new CheckBox { Text = "Subscribe to newsletter", Left = 140, Top = 140, Width = 240 }; p1.Controls.Add(chk);
        var gb = new GroupBox { Text = "Size", Left = 140, Top = 175, Width = 260, Height = 56 };
        var rbs = new List<RadioButton>(); int rx = 10;
        foreach (var t in new[] { "Small", "Medium", "Large" }) { var r = new RadioButton { Text = t, Left = rx, Top = 22, Width = 80 }; gb.Controls.Add(r); rbs.Add(r); rx += 82; }
        p1.Controls.Add(gb);
        var bSubmit = B("Submit", 140, 245, 120); p1.Controls.Add(bSubmit);
        var bReset = B("Reset form", 270, 245, 120); p1.Controls.Add(bReset);
        var lStatus = L("Not submitted", 140, 290); p1.Controls.Add(lStatus);
        bSubmit.Click += (o, e) =>
        {
            form["name"] = tName.Text; form["email"] = tEmail.Text; form["country"] = cb.SelectedItem == null ? "" : cb.SelectedItem.ToString();
            form["subscribe"] = chk.Checked; var rb = rbs.FirstOrDefault(r => r.Checked); form["size"] = rb == null ? "" : rb.Text;
            submitted = true; Save(); lStatus.Text = "Submitted: " + tName.Text;
        };
        bReset.Click += (o, e) => { tName.Text = ""; tEmail.Text = ""; cb.SelectedIndex = -1; chk.Checked = false; foreach (var r in rbs) r.Checked = false; lStatus.Text = "Not submitted"; };

        // ---- List
        var p2 = NewTab("List");
        var lb = new ListBox { Left = 20, Top = 20, Width = 240, Height = 160, AccessibleName = "Items" };
        foreach (var it in items) lb.Items.Add(it); p2.Controls.Add(lb);
        var bRem = B("Remove selected", 280, 20, 150); p2.Controls.Add(bRem);
        var bDel = B("Delete all", 280, 60, 150); p2.Controls.Add(bDel);
        var lCount = L("Items: 5", 20, 190); p2.Controls.Add(lCount);
        Action Sync = () => { items = lb.Items.Cast<object>().Select(x => x.ToString()).ToList(); lCount.Text = "Items: " + lb.Items.Count; Save(); };
        bRem.Click += (o, e) => { if (lb.SelectedIndex >= 0) { lb.Items.RemoveAt(lb.SelectedIndex); Sync(); } };
        bDel.Click += (o, e) =>
        {
            if (MessageBox.Show(f, "Delete all items? This cannot be undone.", "Confirm delete", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            { lb.Items.Clear(); deletedAll = true; Sync(); }
        };

        // ---- Composer
        var p3 = NewTab("Composer");
        var tr = new TextBox { Multiline = true, ReadOnly = true, Left = 20, Top = 20, Width = 680, Height = 220, AccessibleName = "Transcript", ScrollBars = ScrollBars.Vertical }; p3.Controls.Add(tr);
        var prompt = T(20, 290, 560, "Prompt"); p3.Controls.Add(prompt);
        var bSend = B("Send", 590, 288, 90); p3.Controls.Add(bSend);
        var sug = new ListBox { Left = 20, Top = 316, Width = 300, Height = 60, Visible = false, AccessibleName = "Suggestions" };
        sug.Items.AddRange(new object[] { "remote-control", "review", "rename" }); p3.Controls.Add(sug); sug.BringToFront();
        Action SendPrompt = () =>
        {
            string t = prompt.Text; if (t.Trim() == "") return;
            tr.AppendText("You: " + t + "\r\n"); sent.Add(t); prompt.Text = ""; sug.Visible = false; Save();
        };
        prompt.TextChanged += (o, e) => { if (prompt.Text.StartsWith("/") && prompt.Text.Length >= 2) { sug.SelectedIndex = 0; sug.Visible = true; } else sug.Visible = false; };
        prompt.KeyDown += (o, e) =>
        {
            if (e.KeyCode == Keys.Return)
            {
                e.SuppressKeyPress = true;
                if (sug.Visible) { prompt.Text = sug.SelectedItem.ToString() + " "; prompt.SelectionStart = prompt.Text.Length; sug.Visible = false; }   // the autocomplete takes the Enter
                else SendPrompt();
            }
            else if (e.KeyCode == Keys.Escape) { sug.Visible = false; e.SuppressKeyPress = true; }
        };
        bSend.Click += (o, e) => SendPrompt();

        // ---- Dialogs
        var p4 = NewTab("Dialogs");
        var bSet = B("Open settings", 20, 20, 150); p4.Controls.Add(bSet);
        var bNote = B("Show notice", 20, 60, 150); p4.Controls.Add(bNote);
        p4.Controls.Add(L("Notes", 20, 110));
        var notes = new TextBox { Multiline = true, Left = 20, Top = 130, Width = 560, Height = 200, AccessibleName = "Notes", ScrollBars = ScrollBars.Vertical }; p4.Controls.Add(notes);
        var bExp = B("Export...", 20, 340, 150); p4.Controls.Add(bExp);
        var lSaved = L("Saved settings: none", 200, 24); p4.Controls.Add(lSaved);   // what the Settings dialog saved (it reopens with these)
        notes.TextChanged += (o, e) => { notesText = notes.Text; Save(); };
        bNote.Click += (o, e) => { noticeShown++; Save(); MessageBox.Show(f, "Heads up: this is a notice.", "Notice", MessageBoxButtons.OK, MessageBoxIcon.Information); };
        bExp.Click += (o, e) =>
        {
            var d = new SaveFileDialog { InitialDirectory = root, Filter = "Text files (*.txt)|*.txt", FileName = "notes.txt" };
            if (d.ShowDialog(f) == DialogResult.OK) { File.WriteAllText(d.FileName, notes.Text, new UTF8Encoding(false)); savedFiles.Add(d.FileName); Save(); }
        };
        bSet.Click += (o, e) =>
        {
            var dlg = new Form { Text = "Settings", Width = 380, Height = 230, StartPosition = FormStartPosition.CenterParent, FormBorderStyle = FormBorderStyle.FixedDialog, MaximizeBox = false, MinimizeBox = false };
            dlg.Controls.Add(L("Display name", 20, 24)); var dn = T(120, 20, 200, "Display name"); dlg.Controls.Add(dn);
            var dk = new CheckBox { Text = "Dark mode", Left = 120, Top = 60 }; dlg.Controls.Add(dk);
            dn.Text = (string)settings["display_name"]; dk.Checked = (bool)settings["dark"];
            var ok = B("OK", 120, 110, 80); ok.DialogResult = DialogResult.OK; var ca = B("Cancel", 210, 110, 80); ca.DialogResult = DialogResult.Cancel;
            dlg.Controls.Add(ok); dlg.Controls.Add(ca); dlg.AcceptButton = ok; dlg.CancelButton = ca;
            if (dlg.ShowDialog(f) == DialogResult.OK) { settings["display_name"] = dn.Text; settings["dark"] = dk.Checked; settings["saved"] = true; Save();
                lSaved.Text = "Saved settings: name " + dn.Text + ", dark mode " + (dk.Checked ? "on" : "off"); }
        };

        // ---- Login
        var p5 = NewTab("Login");
        p5.Controls.Add(L("Username", 20, 24)); var uBox = T(140, 20, 200, "Username"); p5.Controls.Add(uBox);
        p5.Controls.Add(L("Password", 20, 64)); var pBox = T(140, 60, 200, "Password"); pBox.UseSystemPasswordChar = true; p5.Controls.Add(pBox);
        var bLogin = B("Sign in", 140, 100, 100); p5.Controls.Add(bLogin);
        var lLogin = L("Not signed in", 140, 140); p5.Controls.Add(lLogin);
        bLogin.Click += (o, e) =>
        {
            signedIn = uBox.Text.Length > 0 && pBox.Text.Length > 0;
            lLogin.Text = signedIn ? "Signed in as " + uBox.Text : "Enter a username and a password";
            Save();
        };
        pBox.TextChanged += (o, e) => { if (pBox.Text.Length > 0) { passwordTyped = true; Save(); } };
        uBox.TextChanged += (o, e) => { username = uBox.Text; Save(); };

        // ---- Counter
        var p6 = NewTab("Counter");
        var bInc = B("Increment", 20, 20, 120); p6.Controls.Add(bInc);
        var lCnt = L("Count: 0", 20, 64); lCnt.Font = new Font("Segoe UI", 16); p6.Controls.Add(lCnt);
        bInc.Click += (o, e) => { count++; lCnt.Text = "Count: " + count; Save(); };

        f.FormClosing += (o, e) => { started = false; Save(); };
        Save();
        Application.Run(f);
    }
}
