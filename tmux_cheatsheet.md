# 🧠 Tmux Cheat Sheet

## 🧭 Prefix Key
Most commands start with the **prefix**:
```
Ctrl + b
```
(You can change this to something else, e.g., `Ctrl + a`, in your `~/.tmux.conf`.)

---

## 🪟 Windows (Tabs)
- **Create new window:** `Ctrl + b` → `c`
- **List windows:** `Ctrl + b` → `w`
- **Next window:** `Ctrl + b` → `n`
- **Previous window:** `Ctrl + b` → `p`
- **Switch to window #i:** `Ctrl + b` → `<number>` (e.g., `Ctrl+b 1`)
- **Rename window:** `Ctrl + b` → `,`
- **Close current window:** `exit` or `Ctrl + d`

---

## 🔲 Panes (Splits)
- **Split vertically:** `Ctrl + b` → `%`
- **Split horizontally:** `Ctrl + b` → `"`
- **Switch panes:** `Ctrl + b` → arrow keys (`←`, `→`, etc.)
- **Resize pane:** `Ctrl + b` → hold `Ctrl` and use arrow keys (or `Alt` + arrows if configured)
- **Swap panes:** `Ctrl + b` → `{` or `}`
- **Close pane:** `exit` or `Ctrl + d`

---

## 📜 Scrolling and Copy Mode
- **Enter scroll (copy) mode:** `Ctrl + b` → `[`  
  - Use **↑ / ↓ / PgUp / PgDn** to move around.  
  - Press **q** or **Esc** to exit copy mode.
- **Search in scrollback:** `/`, then `Enter`
- **Copy selection:**  
  - Press **Space** to start selection, move cursor, then **Enter** to copy.
- **Paste copied text:** `Ctrl + b` → `]`

---

## 💡 Session Management
- **Detach from session:** `Ctrl + b` → `d`
- **List sessions:** `tmux ls`
- **Attach to session:** `tmux attach -t <name>`
- **New named session:** `tmux new -s <name>`
- **Kill session:** `tmux kill-session -t <name>`

---

## ⚙️ Bonus / Quality of Life Tips
- **Enable mouse support:**
  ```bash
  set -g mouse on
  ```
- **Use Vi-style copy mode:**
  ```bash
  setw -g mode-keys vi
  ```
- **Reload tmux config:**
  ```bash
  Ctrl + b :source-file ~/.tmux.conf
  ```

---

## 🧩 Hierarchy
```
Session
 ├── Window 0
 │    ├── Pane 0
 │    └── Pane 1
 └── Window 1
      └── Pane 0
```
