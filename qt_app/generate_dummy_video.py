import cv2
import numpy as np
import os
import math

FPS = 30.0
DURATION_S = 20
WIDTH = 800
HEIGHT = 600
ARENA_PADDING = 40

def generate_dummy_video(output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_behavior_test.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = int(DURATION_S * FPS)
    arena_tl = (ARENA_PADDING, ARENA_PADDING)
    arena_br = (WIDTH - ARENA_PADDING, HEIGHT - ARENA_PADDING)
    arena_w = arena_br[0] - arena_tl[0]
    arena_h = arena_br[1] - arena_tl[1]
    zone_split = arena_tl[0] + arena_w // 2

    bg_color = (220, 210, 190)
    light_zone_color = (245, 235, 200)
    dark_zone_color = (180, 160, 130)
    arena_border_color = (80, 80, 80)

    mouse_color_bgr = (30, 30, 30)
    mouse_radius = 12

    np.random.seed(42)
    cx = arena_tl[0] + 100
    cy = arena_tl[1] + 100
    angle = 0.0
    speed = 2.5

    object_center = (arena_br[0] - 120, arena_tl[1] + arena_h // 2)
    object_radius = 35

    print(f"Generating {DURATION_S}s dummy video ({total_frames} frames)...")
    print(f"  Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"  Output: {output_path}")

    for frame_idx in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        cv2.rectangle(frame, arena_tl, arena_br, light_zone_color, -1)
        cv2.rectangle(frame, (zone_split, arena_tl[1]), arena_br, dark_zone_color, -1)
        cv2.line(frame, (zone_split, arena_tl[1]), (zone_split, arena_br[1]), (120, 120, 120), 2)
        cv2.rectangle(frame, arena_tl, arena_br, arena_border_color, 3)

        angle += 0.03 + 0.02 * math.sin(frame_idx * 0.05)
        speed_var = speed + 1.8 * math.sin(frame_idx * 0.02)
        cx += math.cos(angle) * speed_var
        cy += math.sin(angle) * speed_var

        margin = mouse_radius + 5
        cx = max(arena_tl[0] + margin, min(arena_br[0] - margin, cx))
        cy = max(arena_tl[1] + margin, min(arena_br[1] - margin, cy))

        if cx <= arena_tl[0] + margin + 1 or cx >= arena_br[0] - margin - 1:
            angle = math.pi - angle
        if cy <= arena_tl[1] + margin + 1 or cy >= arena_br[1] - margin - 1:
            angle = -angle

        cv2.circle(frame, object_center, object_radius, (100, 70, 150), -1)
        cv2.circle(frame, object_center, object_radius, (60, 40, 100), 2)
        cv2.putText(frame, "OBJ1", (object_center[0] - 18, object_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mx, my = int(cx), int(cy)
        cv2.ellipse(frame, (mx, my), (mouse_radius, int(mouse_radius * 0.75)),
                     angle * 180 / math.pi, 0, 360, mouse_color_bgr, -1)
        tail_end = (int(mx - math.cos(angle) * mouse_radius * 1.5),
                   int(my - math.sin(angle) * mouse_radius * 1.5))
        cv2.line(frame, (mx, my), tail_end, mouse_color_bgr, 3)

        cv2.putText(frame, f"Frame {frame_idx + 1}/{total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        t_seconds = frame_idx / FPS
        cv2.putText(frame, f"T={t_seconds:.1f}s", (WIDTH - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

        if (frame_idx + 1) % int(FPS * 5) == 0:
            print(f"  ... {frame_idx + 1}/{total_frames} frames ({100 * (frame_idx + 1) / total_frames:.0f}%)")

    out.release()
    print(f"\nDone. Video saved to: {output_path}")
    print(f"\nSuggested setup values for testing:")
    print(f"  - Zone names (example): Light,Dark")
    print(f"    Light zone = left half (warmer, brighter color)")
    print(f"    Dark zone = right half (cooler, darker color)")
    print(f"  - Object name (example): Cup")
    print(f"    Purple circle on the right half, near center vertically")
    print(f"  - Detection threshold: ~25 works well for this synthetic video")
    return output_path


if __name__ == "__main__":
    generate_dummy_video()
