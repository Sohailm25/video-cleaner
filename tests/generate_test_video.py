"""Generate a synthetic screen recording with PII for testing."""

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 30, 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_recording.mp4", fourcc, FPS, (WIDTH, HEIGHT))

font = cv2.FONT_HERSHEY_SIMPLEX
font_sm = cv2.FONT_HERSHEY_PLAIN

# Frames 0-89 (0-3s): CRM dashboard with patient info
# Frames 90-179 (3-6s): Email client
# Frames 180-299 (6-10s): Terminal with secrets

for i in range(FPS * DURATION):
    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 245  # light gray bg

    # Top bar (always visible)
    cv2.rectangle(frame, (0, 0), (WIDTH, 50), (50, 50, 50), -1)
    cv2.putText(frame, "MedTrack Pro v3.2.1", (10, 35), font, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, "Dashboard", (250, 35), font_sm, 1.2, (180, 180, 180), 1)
    cv2.putText(frame, "Reports", (360, 35), font_sm, 1.2, (180, 180, 180), 1)
    cv2.putText(frame, "Settings", (450, 35), font_sm, 1.2, (180, 180, 180), 1)

    sec = i / FPS

    if sec < 3:
        # CRM / Patient dashboard
        cv2.rectangle(frame, (20, 70), (400, 350), (255, 255, 255), -1)
        cv2.rectangle(frame, (20, 70), (400, 110), (70, 130, 180), -1)
        cv2.putText(frame, "Patient Details", (30, 98), font, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, "Name: Sarah Johnson", (30, 140), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "DOB: 03/15/1987", (30, 165), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "MRN: 4829173", (30, 190), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "SSN: 412-55-7890", (30, 215), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Phone: (555) 234-5678", (30, 240), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Email: sarah.johnson@email.com", (30, 265), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Address: 1234 Oak Drive", (30, 290), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Insurance ID: BC-88291034", (30, 315), font_sm, 1.2, (30, 30, 30), 1)

        # Right panel - notes
        cv2.rectangle(frame, (420, 70), (800, 350), (255, 255, 255), -1)
        cv2.rectangle(frame, (420, 70), (800, 110), (70, 130, 180), -1)
        cv2.putText(frame, "Visit Notes", (430, 98), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Patient reports recurring headaches", (430, 140), font_sm, 1.1, (60, 60, 60), 1)
        cv2.putText(frame, "since January. Prescribed ibuprofen.", (430, 162), font_sm, 1.1, (60, 60, 60), 1)
        cv2.putText(frame, "Follow-up in 2 weeks.", (430, 184), font_sm, 1.1, (60, 60, 60), 1)

    elif sec < 6:
        # Email client view
        cv2.rectangle(frame, (20, 70), (900, 400), (255, 255, 255), -1)
        cv2.rectangle(frame, (20, 70), (900, 110), (180, 70, 70), -1)
        cv2.putText(frame, "Inbox - Outlook", (30, 98), font, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, "From: mike.chen@acmecorp.com", (30, 140), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "To: sarah.johnson@email.com", (30, 165), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Subject: Q4 Performance Review", (30, 190), font_sm, 1.2, (30, 30, 30), 1)
        cv2.putText(frame, "Hi Sarah,", (30, 225), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "Your employee ID is EMP-44291. Please review", (30, 250), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "the attached document and sign by Friday.", (30, 275), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "Credit card ending 4532-8821-0099-7766", (30, 300), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "was charged $250.00 for the conference.", (30, 325), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "Best, Mike", (30, 360), font_sm, 1.2, (60, 60, 60), 1)
        cv2.putText(frame, "IP: 192.168.1.105", (30, 385), font_sm, 1.0, (150, 150, 150), 1)

    else:
        # Terminal / code editor with secrets
        cv2.rectangle(frame, (20, 70), (900, 500), (30, 30, 30), -1)
        green = (0, 200, 0)
        white = (200, 200, 200)
        cv2.putText(frame, "$ cat .env", (30, 100), font_sm, 1.2, green, 1)
        cv2.putText(frame, "API_KEY=sk-proj-a8Bf92kLmN3pQrStUvWx", (30, 130), font_sm, 1.2, white, 1)
        cv2.putText(frame, "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE", (30, 155), font_sm, 1.2, white, 1)
        cv2.putText(frame, "DATABASE_URL=postgres://admin:s3cretPwd@db.internal:5432", (30, 180), font_sm, 1.2, white, 1)
        cv2.putText(frame, "SECRET_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abc", (30, 205), font_sm, 1.2, white, 1)
        cv2.putText(frame, "PASSWORD=hunter2supersecret", (30, 230), font_sm, 1.2, white, 1)
        cv2.putText(frame, "$ ssh admin@10.0.0.42", (30, 270), font_pm if False else font_sm, 1.2, green, 1)
        cv2.putText(frame, "Connection established to 10.0.0.42", (30, 295), font_sm, 1.2, white, 1)

    # Status bar at bottom
    cv2.rectangle(frame, (0, HEIGHT - 30), (WIDTH, HEIGHT), (60, 60, 60), -1)
    cv2.putText(frame, f"User: sjohnson | Session: {i//FPS}s", (10, HEIGHT - 10), font_sm, 1.0, (180, 180, 180), 1)

    out.write(frame)

out.release()
print(f"Created test_recording.mp4: {WIDTH}x{HEIGHT}, {FPS}fps, {DURATION}s")
