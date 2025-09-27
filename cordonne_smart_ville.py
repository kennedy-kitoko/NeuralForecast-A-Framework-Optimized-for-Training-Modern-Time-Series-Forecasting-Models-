import json
import math
import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import cv2
import numpy as np

# -------------------- ÉTAT GLOBAL --------------------
image_bgr = None          # image originale (BGR OpenCV)
clone_bgr = None
display_scale = 1.0       # facteur d'échelle (affichage Tkinter)
max_display_w = 1280
max_display_h = 800

current_shape = "polygon" # polygon | circle | rectangle | line | ring
current_pts = []          # points en cours (coordonnées 'image')
cursor_xy = None          # dernière position curseur (coord 'image')

shapes = []               # formes validées
current_color = (255, 0, 0)  # BGR
img_label = None          # widget pour l'image
right_panel = None

# -------------------- OUTILS ÉCHELLE --------------------
def to_img_xy(ev_x, ev_y):
    """Convertit coord Tkinter (affichage) -> coord image originale."""
    x = int(ev_x / display_scale)
    y = int(ev_y / display_scale)
    return x, y

def scale_bgr_for_display(bgr):
    """Redimensionne pour l'affichage + met à jour display_scale."""
    global display_scale
    h, w = bgr.shape[:2]
    sx = max_display_w / w
    sy = max_display_h / h
    s = min(sx, sy, 1.0)
    display_scale = s
    if s < 1.0:
        new_size = (int(w*s), int(h*s))
        return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    return bgr

# -------------------- OUVRIR / SAUVER --------------------
def open_image():
    global image_bgr, clone_bgr, current_pts, shapes, cursor_xy
    path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
    if not path:
        return
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        print("Impossible d'ouvrir l'image.")
        return
    clone_bgr = image_bgr.copy()
    shapes.clear()
    current_pts.clear()
    cursor_xy = None
    update_display()

def save_json():
    if not shapes:
        print("Aucune forme à sauvegarder.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
    if not path:
        return
    data = []
    for i, sh in enumerate(shapes, start=1):
        sh_copy = dict(sh)
        sh_copy["id"] = i
        sh_copy["name"] = sh_copy.get("name", f"Zone {i}")
        if sh_copy["type"] == "polygon":
            sh_copy["points"] = [list(p) for p in sh_copy["points"]]
        elif sh_copy["type"] == "circle":
            sh_copy["center"] = list(sh_copy["center"])
        elif sh_copy["type"] == "rectangle":
            sh_copy["p1"] = list(sh_copy["p1"])
            sh_copy["p2"] = list(sh_copy["p2"])
        elif sh_copy["type"] == "line":
            sh_copy["p1"] = list(sh_copy["p1"])
            sh_copy["p2"] = list(sh_copy["p2"])
        elif sh_copy["type"] == "ring":
            sh_copy["center"] = list(sh_copy["center"])
        sh_copy["color"] = list(sh_copy["color"])  # BGR
        data.append(sh_copy)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Sauvé : {path}")

# -------------------- DESSIN FORMES --------------------
def draw_filled_and_outline(img, pts, fill_color, outline_color=(0,0,0), thickness=2):
    if len(pts) >= 3:
        cv2.fillPoly(img, [np.array(pts, np.int32)], fill_color)
    cv2.polylines(img, [np.array(pts, np.int32)], False, outline_color, thickness, lineType=cv2.LINE_AA)

def draw_shape(img, sh, label_text=None, preview=False):
    color = (0,0,0) if preview else sh["color"]
    outline = (0,0,0)
    if sh["type"] == "polygon":
        pts = sh["points"]
        if preview:
            cv2.polylines(img, [np.array(pts, np.int32)], False, outline, 2, lineType=cv2.LINE_AA)
            if "cursor" in sh and sh["cursor"] is not None and len(pts) >= 1:
                cv2.line(img, pts[-1], sh["cursor"], outline, 2, lineType=cv2.LINE_AA)
        else:
            draw_filled_and_outline(img, pts, color, outline)
            if len(pts) >= 3:
                M = cv2.moments(np.array(pts, np.int32))
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(img, label_text or "", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)

    elif sh["type"] == "circle":
        c, r = sh["center"], sh["radius"]
        if preview:
            cv2.circle(img, c, r, outline, 2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, c, r, outline, 2, lineType=cv2.LINE_AA)
            cv2.putText(img, label_text or "", c, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)

    elif sh["type"] == "rectangle":
        p1, p2 = sh["p1"], sh["p2"]
        if preview:
            cv2.rectangle(img, p1, p2, outline, 2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(img, p1, p2, color, -1, lineType=cv2.LINE_AA)
            cv2.rectangle(img, p1, p2, outline, 2, lineType=cv2.LINE_AA)
            cx = (p1[0]+p2[0])//2; cy = (p1[1]+p2[1])//2
            cv2.putText(img, label_text or "", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)

    elif sh["type"] == "line":
        p1, p2 = sh["p1"], sh["p2"]
        col = outline if preview else color
        cv2.line(img, p1, p2, col, 3, lineType=cv2.LINE_AA)
        if not preview:
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(img, label_text or "", mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)

    elif sh["type"] == "ring":
        c = sh["center"]
        rin = sh["inner_radius"]
        rout = sh["outer_radius"]
        if preview:
            cv2.circle(img, c, rin, outline, 2, lineType=cv2.LINE_AA)
            if "outer_preview" in sh and sh["outer_preview"] is not None:
                cv2.circle(img, c, sh["outer_preview"], outline, 2, lineType=cv2.LINE_AA)
        else:
            mask = np.zeros_like(img)
            cv2.circle(mask, c, rout, sh["color"], -1, lineType=cv2.LINE_AA)
            cv2.circle(mask, c, rin, (0,0,0), -1, lineType=cv2.LINE_AA)
            img[:] = cv2.addWeighted(img, 1, mask, 1, 0)
            cv2.circle(img, c, rout, outline, 2, lineType=cv2.LINE_AA)
            cv2.circle(img, c, rin, outline, 2, lineType=cv2.LINE_AA)
            cv2.putText(img, label_text or "", c, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)

# -------------------- AFFICHAGE --------------------
def update_display():
    if image_bgr is None:
        img_label.config(image="", text="Ouvre une image…")
        return

    base = clone_bgr.copy()
    overlay = base.copy()
    for i, sh in enumerate(shapes, start=1):
        sh_lbl = f"Zone {i}"
        draw_shape(overlay, sh, sh_lbl, preview=False)

    temp = build_preview_shape()
    if temp is not None:
        draw_shape(overlay, temp, preview=True)

    composed = cv2.addWeighted(overlay, 0.4, base, 0.6, 0)
    disp = scale_bgr_for_display(composed)

    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(disp_rgb)
    tk_im = ImageTk.PhotoImage(image=im)

    img_label.config(image=tk_im, text="", width=disp.shape[1], height=disp.shape[0])
    img_label.image = tk_im

def build_preview_shape():
    if image_bgr is None or not current_pts:
        return None
    if current_shape == "polygon":
        return {"type":"polygon", "points": current_pts.copy(), "color": (0,0,0), "cursor": cursor_xy}
    elif current_shape == "circle" and len(current_pts) == 1 and cursor_xy is not None:
        r = int(math.hypot(cursor_xy[0]-current_pts[0][0], cursor_xy[1]-current_pts[0][1]))
        return {"type":"circle","center":current_pts[0],"radius":r,"color":(0,0,0)}
    elif current_shape == "rectangle" and len(current_pts) == 1 and cursor_xy is not None:
        return {"type":"rectangle","p1":current_pts[0],"p2":cursor_xy,"color":(0,0,0)}
    elif current_shape == "line" and len(current_pts) == 1 and cursor_xy is not None:
        return {"type":"line","p1":current_pts[0],"p2":cursor_xy,"color":(0,0,0)}
    elif current_shape == "ring":
        if len(current_pts) == 1 and cursor_xy is not None:
            r_in = int(math.hypot(cursor_xy[0]-current_pts[0][0], cursor_xy[1]-current_pts[0][1]))
            return {"type":"ring","center":current_pts[0],"inner_radius":r_in,"outer_radius":r_in,"outer_preview":None,"color":(0,0,0)}
        if len(current_pts) == 2 and cursor_xy is not None:
            r_in = int(math.hypot(current_pts[1][0]-current_pts[0][0], current_pts[1][1]-current_pts[0][1]))
            r_out = int(math.hypot(cursor_xy[0]-current_pts[0][0], cursor_xy[1]-current_pts[0][1]))
            return {"type":"ring","center":current_pts[0],"inner_radius":r_in,"outer_radius":r_out,"outer_preview":r_out,"color":(0,0,0)}
    return None

# -------------------- INTERACTIONS SOURIS --------------------
def on_mouse_move(event):
    global cursor_xy
    if image_bgr is None:
        return
    cursor_xy = to_img_xy(event.x, event.y)
    update_display()

def on_left_click(event):
    if image_bgr is None:
        return
    x, y = to_img_xy(event.x, event.y)
    if current_shape in ("circle", "rectangle", "line"):
        if len(current_pts) == 0:
            current_pts.append((x, y))
    elif current_shape == "polygon":
        current_pts.append((x, y))
    elif current_shape == "ring":
        if len(current_pts) < 2:
            current_pts.append((x, y))
    update_display()

def on_right_click(event):
    global current_pts
    if image_bgr is None:
        return
    x, y = to_img_xy(event.x, event.y)
    if current_shape == "polygon":
        if len(current_pts) >= 3:
            shapes.append({"type":"polygon","points":current_pts.copy(),"color":current_color})
        current_pts = []
    elif current_shape == "circle":
        if len(current_pts) == 1:
            r = int(math.hypot(x-current_pts[0][0], y-current_pts[0][1]))
            shapes.append({"type":"circle","center":current_pts[0],"radius":r,"color":current_color})
            current_pts = []
    elif current_shape == "rectangle":
        if len(current_pts) == 1:
            shapes.append({"type":"rectangle","p1":current_pts[0],"p2":(x,y),"color":current_color})
            current_pts = []
    elif current_shape == "line":
        if len(current_pts) == 1:
            shapes.append({"type":"line","p1":current_pts[0],"p2":(x,y),"color":current_color})
            current_pts = []
    elif current_shape == "ring":
        if len(current_pts) == 2:
            center = current_pts[0]
            r_in = int(math.hypot(current_pts[1][0]-center[0], current_pts[1][1]-center[1]))
            r_out = int(math.hypot(x-center[0], y-center[1]))
            if r_out <= r_in:
                r_out = r_in + 1
            shapes.append({"type":"ring","center":center,"inner_radius":r_in,"outer_radius":r_out,"color":current_color})
            current_pts = []
    update_display()

# -------------------- COMMANDES UI --------------------
def set_shape(shape):
    global current_shape, current_pts
    current_shape = shape
    current_pts = []
    print(f"Forme: {shape}")

def choose_color():
    global current_color
    c = colorchooser.askcolor(title="Choisir une couleur")[0]
    if c:
        current_color = (int(c[2]), int(c[1]), int(c[0]))

def undo_last():
    if shapes:
        shapes.pop()
        update_display()

def cancel_current():
    current_pts.clear()
    update_display()

def clear_all():
    shapes.clear()
    current_pts.clear()
    update_display()

# -------------------- UI --------------------
root = tk.Tk()
root.title("Annotateur de zones routières – MTMC")

# Zone image (gauche)
img_label = tk.Label(root, text="Ouvre une image…", bg="#222", fg="#ddd")
img_label.pack(side="left")  # ⚠️ pas expand/fill pour garder la taille exacte
img_label.bind("<Motion>", on_mouse_move)
img_label.bind("<Button-1>", on_left_click)
img_label.bind("<Button-3>", on_right_click)

# Panneau de droite (boutons en colonne)
right_panel = tk.Frame(root)
right_panel.pack(side="right", fill="y", padx=8, pady=8)

tk.Button(right_panel, text="Ouvrir image", command=open_image, width=18).pack(pady=4)
tk.Button(right_panel, text="Sauver JSON", command=save_json, width=18).pack(pady=4)

tk.Label(right_panel, text="Formes", font=("Arial", 10, "bold")).pack(pady=(10,2))
for s in ("polygon","circle","rectangle","line","ring"):
    tk.Button(right_panel, text=s.capitalize(), command=lambda sh=s: set_shape(sh), width=18).pack(pady=2)

tk.Button(right_panel, text="Choisir couleur", command=choose_color, width=18).pack(pady=(10,4))

tk.Label(right_panel, text="Édition", font=("Arial", 10, "bold")).pack(pady=(10,2))
tk.Button(right_panel, text="↶ Undo (dernière forme)", command=undo_last, width=18).pack(pady=2)
tk.Button(right_panel, text="✖ Cancel (forme en cours)", command=cancel_current, width=18).pack(pady=2)
tk.Button(right_panel, text="🗑 Clear tout", command=clear_all, width=18).pack(pady=2)

root.mainloop()
