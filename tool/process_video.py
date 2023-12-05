import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import numpy as np

def apply_time_bending_effect(input_file, output_file):
    clip = VideoFileClip(input_file)
    duration = clip.duration
    segment_duration = duration / 7
    fps = clip.fps

    clips = []
    for i in range(7):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = clip.subclip(start_time, end_time)

        if segment.duration is not None and segment.duration > 0:
            t = i / 6
            speed_factor = 1 + 0.5 * np.sin(np.pi * t)

            new_duration = segment.duration / speed_factor
            if new_duration is not None and new_duration > 0:
                segment = segment.fx(vfx.speedx, speed_factor)
                segment = segment.set_duration(new_duration)

                clips.append(segment)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", fps=fps)

def main():
    input_folder = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/asub_FALL_slow"
    output_folder = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/FALL"

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            apply_time_bending_effect(input_file, output_file)

if __name__ == "__main__":
    main()
