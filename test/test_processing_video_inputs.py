from __future__ import annotations

import unittest

from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    VideoInputSpec,
    build_video_messages,
)


class ProcessingVideoInputsTest(unittest.TestCase):
    def test_frame_paths_build_qwen_video_frame_list_message(self) -> None:
        messages = build_video_messages(
            VideoInputSpec(
                frame_paths=["/tmp/f0.jpg", "/tmp/f1.jpg"],
                sample_fps=1,
            ),
            prompt="describe",
        )

        video_payload = messages[0]["content"][0]
        self.assertEqual(video_payload["type"], "video")
        self.assertEqual(video_payload["video"], ["file:///tmp/f0.jpg", "file:///tmp/f1.jpg"])
        self.assertEqual(video_payload["sample_fps"], 1)
        self.assertEqual(messages[0]["content"][1]["text"], "describe")

    def test_video_path_builds_qwen_full_video_message(self) -> None:
        messages = build_video_messages(
            VideoInputSpec(
                video_path="/tmp/sample.mp4",
                video_fps=2.0,
                video_start=1.0,
                video_end=3.0,
                video_min_frames=4,
                video_max_frames=32,
            ),
            prompt="describe",
        )

        video_payload = messages[0]["content"][0]
        self.assertEqual(video_payload["type"], "video")
        self.assertEqual(video_payload["video"], "/tmp/sample.mp4")
        self.assertEqual(video_payload["fps"], 2.0)
        self.assertEqual(video_payload["video_start"], 1.0)
        self.assertEqual(video_payload["video_end"], 3.0)
        self.assertEqual(video_payload["min_frames"], 4)
        self.assertEqual(video_payload["max_frames"], 32)
        self.assertNotIn("sample_fps", video_payload)

    def test_video_fps_and_nframes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            build_video_messages(
                VideoInputSpec(
                    video_path="/tmp/sample.mp4",
                    video_fps=2.0,
                    video_nframes=8,
                )
            )


if __name__ == "__main__":
    unittest.main()
