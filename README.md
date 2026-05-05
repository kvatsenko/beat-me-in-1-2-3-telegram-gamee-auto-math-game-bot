# Math Game OCR Bot

A Python bot that automatically detects a math game on the screen, reads the equation with OCR, solves it, and clicks the correct answer button.

The bot was built for a game where the user has to solve simple equations and choose one of three answers: `1`, `2`, or `3`.

## Demo

### OCR Input

This is the cropped and preprocessed equation image that the bot sends to OCR.

![OCR input](debug_capture.png)

### Screen Detection

This debug image shows how the bot detects the game interface.

![Full detection](debug_full_detection.png)

In this image:

- blue rectangle — detected game area;
- green rectangle — detected equation area;
- red dots — detected answer button centers.

### Video Demo

You can add a demonstration video showing how the bot works in real time.

Place your video file in the project folder, for example:

```text
video_demo.mp4
```

Then you have two options:

**1. Embed video (works in some Markdown viewers):**

```html
<video src="video_demo.mp4" controls width="700"></video>
```

**2. Add a simple link (recommended for GitHub):**

[Watch demo video](video_demo.mp4)