# Smart Vertical Video Generator 🎬📱

An AI-powered video reframing system that automatically converts landscape videos (16:9) into vertical short-form content (9:16) for platforms like Instagram Reels, YouTube Shorts, and TikTok.

The system intelligently detects and tracks people in the frame using YOLO and dynamically adjusts the crop area to keep subjects properly visible throughout the video.

---

##  Features

* Automatic 16:9 → 9:16 video conversion
* AI-based person detection using YOLO
* Smart subject tracking and reframing
* Smooth camera movement and crop transitions
* Multi-person handling logic
* Real-time frame processing
* Flask backend integration
* Optimized for short-form content creation

---

## Tech Stack

* Python
* YOLO
* OpenCV
* Flask
* NumPy

---

## How It Works

1. Input a landscape video
2. YOLO detects people in each frame
3. The system calculates the best crop position
4. Frames are dynamically reframed into vertical format
5. Final output video is generated in 9:16 aspect ratio

---

## Demo

The project automatically keeps the main subject centered and visible while converting wide videos into mobile-friendly vertical videos.

Example use cases:

* Instagram Reels
* TikTok videos
* YouTube Shorts
* Podcast clips
* Interview videos
* Cinematic reframing

---

---

## 📂 Project Structure

```bash
Smart-Vertical-Video-Generator/
│
├── app.py
├── requirements.txt
├── static/
├── templates/
├── uploads/
├── outputs/
└── README.md
```

---

## 🔥 Future Improvements

* Face priority tracking
* Object-aware reframing
* GPU optimization
* Real-time live video support
* Audio-based speaker focus
* Better multi-person scene handling

---
