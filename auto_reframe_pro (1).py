"""
╔══════════════════════════════════════════════════════════════╗
║         AUTO REFRAME PRO — Production-Grade Pipeline         ║
║                                                              ║
║  ✅ Kalman Filter         — smooth, predictive tracking      ║
║  ✅ IoU Re-ID             — stable person IDs across frames  ║
║  ✅ Hysteresis Switch     — no mode flicker                  ║
║  ✅ Smart Group Crop      — tight crop for 3+ people         ║
║  ✅ Face-Priority Reframe — head-centred, not body-centred   ║
║  ✅ Pose Headroom         — never cuts off the head          ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ─────────────────────────── CONFIG ────────────────────────────
VIDEO_IN       = "demo.mp4"
VIDEO_OUT      = "output_pro.mp4"
OUT_W, OUT_H   = 1080, 1920
HYSTERESIS     = 8
MAX_IOU_DIST   = 0.3
KALMAN_NOISE_Q = 1e-2
KALMAN_NOISE_R = 1e-1
HEADROOM_PAD   = 0.12
SHOW_DEBUG     = True
# ───────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
#  MEDIAPIPE safe import — works on ALL versions
# ══════════════════════════════════════════════════════════════
_pose_detector = None
_POSE_OK = False
try:
    import mediapipe as mp
    _pose_sol = mp.solutions.pose          # raises AttributeError on bad builds
    _pose_detector = _pose_sol.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _POSE_OK = True
    print("MediaPipe Pose loaded OK")
except AttributeError:
    print("mediapipe.solutions not found (new API build).")
    print("Pose Headroom disabled — using YOLO box top.")
    print("Fix: pip install 'mediapipe==0.10.9'")
except Exception as e:
    print(f"MediaPipe failed ({e}) — Pose Headroom disabled.")


# ══════════════════════════════════════════════════════════════
#  1.  KALMAN FILTER
# ══════════════════════════════════════════════════════════════
class KalmanTracker1D:
    def __init__(self, init_val):
        self.kf = cv.KalmanFilter(2, 1)
        self.kf.transitionMatrix    = np.array([[1,1],[0,1]], np.float32)
        self.kf.measurementMatrix   = np.array([[1,0]], np.float32)
        self.kf.processNoiseCov     = np.eye(2, dtype=np.float32) * KALMAN_NOISE_Q
        self.kf.measurementNoiseCov = np.eye(1, dtype=np.float32) * KALMAN_NOISE_R
        self.kf.errorCovPost        = np.eye(2, dtype=np.float32)
        self.kf.statePost           = np.array([[init_val],[0]], dtype=np.float32)

    def update(self, m):
        self.kf.predict()
        return float(self.kf.correct(np.array([[m]], dtype=np.float32))[0][0])

    def predict(self):
        return float(self.kf.predict()[0][0])


# ══════════════════════════════════════════════════════════════
#  2.  IoU RE-ID
# ══════════════════════════════════════════════════════════════
def iou(a, b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter  = max(0,xB-xA)*max(0,yB-yA)
    if not inter: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

class ReIDTracker:
    def __init__(self):
        self.tracks = {}
        self._nid   = 0

    def update(self, dets):
        if not dets:
            for tid in list(self.tracks):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > 10: del self.tracks[tid]
            return []
        used, result = set(), []
        for det in dets:
            best_tid, best_sc = None, MAX_IOU_DIST
            for tid, tr in self.tracks.items():
                if tid in used: continue
                sc = iou(det, tr['box'])
                if sc > best_sc: best_sc, best_tid = sc, tid
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            if best_tid is not None:
                tr = self.tracks[best_tid]
                scx = tr['kx'].update(cx)
                scy = tr['ky'].update(cy)
                tr['box'], tr['age'] = det, 0
                used.add(best_tid)
                result.append((best_tid, det, scx, scy))
            else:
                tid = self._nid; self._nid += 1
                self.tracks[tid] = {'box':det,'kx':KalmanTracker1D(cx),'ky':KalmanTracker1D(cy),'age':0}
                self.tracks[tid]['kx'].update(cx); self.tracks[tid]['ky'].update(cy)
                used.add(tid)
                result.append((tid, det, cx, cy))
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > 10: del self.tracks[tid]
        return result


# ══════════════════════════════════════════════════════════════
#  3.  HYSTERESIS SWITCH
# ══════════════════════════════════════════════════════════════
class HysteresisSwitch:
    def __init__(self, hold=HYSTERESIS):
        self.cur, self.pend, self.cnt, self.hold = None, None, 0, hold

    def get_mode(self, raw):
        if raw == self.cur:
            self.pend, self.cnt = None, 0
            return self.cur
        if raw == self.pend:
            self.cnt += 1
            if self.cnt >= self.hold:
                self.cur, self.pend, self.cnt = raw, None, 0
        else:
            self.pend, self.cnt = raw, 1
        return self.cur if self.cur else raw


# ══════════════════════════════════════════════════════════════
#  4.  FACE DETECTOR
# ══════════════════════════════════════════════════════════════
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")

def face_cx_in_box(frame, x1, y1, x2, y2):
    roi = frame[max(0,y1):y2, max(0,x1):x2]
    if roi.size == 0: return None
    faces = face_cascade.detectMultiScale(cv.cvtColor(roi,cv.COLOR_BGR2GRAY),1.1,4,minSize=(30,30))
    if not len(faces): return None
    fx,fy,fw,fh = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)[0]
    return x1+fx+fw//2


# ══════════════════════════════════════════════════════════════
#  5.  POSE HEADROOM
# ══════════════════════════════════════════════════════════════
def head_top_y(frame, x1, y1, x2, y2):
    if not _POSE_OK or _pose_detector is None: return y1
    roi = frame[max(0,y1):y2, max(0,x1):x2]
    if roi.size == 0: return y1
    try:
        res = _pose_detector.process(cv.cvtColor(roi,cv.COLOR_BGR2RGB))
    except: return y1
    if not res.pose_landmarks: return y1
    h = roi.shape[0]
    ys = [res.pose_landmarks.landmark[i].y*h for i in [0,7,8]
          if res.pose_landmarks.landmark[i].visibility > 0.3]
    return (y1+int(min(ys))) if ys else y1


# ══════════════════════════════════════════════════════════════
#  CROP HELPERS
# ══════════════════════════════════════════════════════════════
def clamp(left, right, fw, cw):
    if left < 0:    left, right = 0, cw
    if right > fw:  right, left = fw, fw-cw
    return max(0,left), right

def crop_9_16(frame, cx):
    fh, fw = frame.shape[:2]
    cw = int(fh*9/16)
    l, r = clamp(cx-cw//2, cx+cw//2, fw, cw)
    return cv.resize(frame[:,l:r], (OUT_W,OUT_H))

def split_two(frame, ba, bb, cxa, cxb):
    fh, fw = frame.shape[:2]
    cw, hw = int(fh*9/16), OUT_W//2
    panels = []
    for (x1,y1,x2,y2), cx in sorted(zip([ba,bb],[cxa,cxb]), key=lambda t:t[0][0]):
        l, r = clamp(int(cx)-cw//2, int(cx)+cw//2, fw, cw)
        panels.append(cv.resize(frame[:,l:r],(hw,OUT_H)))
    out = cv.hconcat(panels)
    cv.line(out,(hw,0),(hw,OUT_H),(200,200,200),2)
    return out

def smart_group(frame, tracked):
    fh, fw = frame.shape[:2]
    ax1=min(p[1][0] for p in tracked); ay1=min(p[1][1] for p in tracked)
    ax2=max(p[1][2] for p in tracked); ay2=max(p[1][3] for p in tracked)
    gw,gh,gcx,gcy = ax2-ax1, ay2-ay1, (ax1+ax2)//2, (ay1+ay2)//2
    cw  = max(gw, int(gh*9/16))
    ch  = int(cw*16/9)
    l,r = clamp(gcx-cw//2, gcx+cw//2, fw, cw)
    t   = max(0, gcy-ch//2); b = min(fh, t+ch)
    sub = frame[t:b, l:r]
    if sub.size == 0: sub = frame
    sh, sw = sub.shape[:2]
    ar = sw/sh if sh else 9/16
    tar = 9/16
    if abs(ar-tar)<0.05: return cv.resize(sub,(OUT_W,OUT_H))
    if ar > tar:
        nw,nh = OUT_W, int(OUT_W/ar)
        pad = (OUT_H-nh)//2
        c = np.zeros((OUT_H,OUT_W,3),dtype=np.uint8)
        c[pad:pad+nh,:] = cv.resize(sub,(nw,nh)); return c
    else:
        nh,nw = OUT_H, int(OUT_H*ar)
        pad = (OUT_W-nw)//2
        c = np.zeros((OUT_H,OUT_W,3),dtype=np.uint8)
        c[:,pad:pad+nw] = cv.resize(sub,(nw,nh)); return c

MODE_COLORS = {"FOLLOW":(0,255,80),"SPLIT":(255,165,0),"GROUP":(0,120,255)}

def overlay(f, mode, n):
    cv.rectangle(f,(0,0),(OUT_W,90),(0,0,0),-1)
    cv.putText(f,f"MODE: {mode}   PERSONS: {n}",(20,62),cv.FONT_HERSHEY_SIMPLEX,1.3,MODE_COLORS.get(mode,(255,255,255)),2)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    model = YOLO("yolov8n.pt")
    cap   = cv.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print("Cannot open:", VIDEO_IN); return

    fps = int(cap.get(cv.CAP_PROP_FPS)) or 30
    out = cv.VideoWriter(VIDEO_OUT, cv.VideoWriter_fourcc(*'mp4v'), fps, (OUT_W,OUT_H))

    reid       = ReIDTracker()
    hyst       = HysteresisSwitch()
    fk         = None          # follow Kalman
    fidx       = 0

    print("Processing... press Q to stop\n")
    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1
        fh, fw = frame.shape[:2]

        # YOLO
        boxes = []
        for r in model(frame, verbose=False):
            for b in r.boxes:
                if int(b.cls[0])==0:
                    boxes.append(tuple(map(int,b.xyxy[0])))

        tracked = reid.update(boxes)
        n = len(tracked)

        raw = "FOLLOW" if n<=1 else "SPLIT" if n==2 else "GROUP"
        mode = hyst.get_mode(raw)

        # draw boxes
        pal = [(0,255,80),(255,165,0),(0,120,255),(255,50,200),(0,220,255)]
        for tid,box,scx,scy in tracked:
            c = pal[tid%len(pal)]
            cv.rectangle(frame,(box[0],box[1]),(box[2],box[3]),c,2)
            cv.putText(frame,f"ID{tid}",(box[0],box[1]-8),cv.FONT_HERSHEY_SIMPLEX,.65,c,2)

        # ── FOLLOW ──
        if mode=="FOLLOW":
            if tracked:
                tid,box,scx,scy = tracked[0]
                x1,y1,x2,y2 = box
                fc = face_cx_in_box(frame,x1,y1,x2,y2)
                tcx = fc if fc else int(scx)
                hy  = head_top_y(frame,x1,y1,x2,y2)
                hy  = max(0, hy - int((y2-y1)*HEADROOM_PAD))
                if fk is None: fk = KalmanTracker1D(tcx)
                scx2 = int(fk.update(tcx))
            else:
                scx2 = int(fk.predict()) if fk else fw//2
            fo = crop_9_16(frame, scx2)

        # ── SPLIT ──
        elif mode=="SPLIT":
            fk = None
            if len(tracked)>=2:
                _,ba,cxa,_ = tracked[0]; _,bb,cxb,_ = tracked[1]
                fo = split_two(frame,ba,bb,cxa,cxb)
            else:
                fo = crop_9_16(frame,fw//2)

        # ── GROUP ──
        else:
            fk = None
            fo = smart_group(frame,tracked) if tracked else cv.resize(frame,(OUT_W,OUT_H))

        if SHOW_DEBUG: overlay(fo,mode,n)
        out.write(fo)
        cv.imshow("Auto Reframe PRO", fo)
        if cv.waitKey(1)&0xFF==ord('q'): break
        if fidx%30==0: print(f"  Frame {fidx:>5} | {mode:<6} | persons:{n}")

    cap.release(); out.release(); cv.destroyAllWindows()
    print(f"\nDone! Saved -> {VIDEO_OUT}")

if __name__=="__main__":
    main()
