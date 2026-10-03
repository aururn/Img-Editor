import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.core.inference import GenerationRequest, InferenceService


class StandardGenerationTests(unittest.TestCase):
    def setUp(self):
        self.output = Image.new("RGB", (72, 80))
        self.pipe = Mock(return_value=SimpleNamespace(images=[self.output]))
        self.manager = Mock()
        for name in ("get_txt2img", "get_img2img", "get_inpaint", "get_controlnet"):
            getattr(self.manager, name).return_value = self.pipe
        self.service = InferenceService(self.manager)
        self.service._generator = Mock(return_value="seeded generator")
        self.embeddings = tuple(object() for _ in range(4))
        self.service._encode_long_prompt = Mock(return_value=self.embeddings)

    def request(self, mode, **kwargs):
        return GenerationRequest(
            mode=mode,
            checkpoint="model",
            image=None if mode == "txt2img" else Image.new("RGB", (75, 83)),
            mask=Image.new("L", (75, 83), 255) if mode.startswith("inpaint") else None,
            prompt="positive",
            negative_prompt="negative",
            width=75,
            height=83,
            **kwargs,
        )

    def test_all_modes_preserve_prompt_and_sampling_parameters(self):
        for mode in ("txt2img", "img2img", "inpaint_manual", "inpaint_auto"):
            with self.subTest(mode=mode):
                self.pipe.reset_mock()
                self.service._encode_long_prompt.reset_mock()
                req = self.request(mode)
                self.assertIs(self.service._run_standard(req, 42), self.output)
                self.service._encode_long_prompt.assert_called_once_with(
                    self.pipe, "positive", "negative"
                )
                self.service._generator.assert_called_with(42)
                kwargs = self.pipe.call_args.kwargs
                for key, value in zip(
                    (
                        "prompt_embeds",
                        "negative_prompt_embeds",
                        "pooled_prompt_embeds",
                        "negative_pooled_prompt_embeds",
                    ),
                    self.embeddings,
                ):
                    self.assertIs(kwargs[key], value)
                self.assertEqual(kwargs["guidance_scale"], 7.0)
                self.assertEqual(kwargs["num_inference_steps"], 28)
                self.assertEqual(kwargs["generator"], "seeded generator")
                self.assertEqual(kwargs["callback_on_step_end"], self.service._step_callback)
                self.assertNotIn("control_image", kwargs)
                if mode != "txt2img":
                    self.assertEqual(kwargs["image"].size, (72, 80))
                    self.assertEqual(kwargs["strength"], 0.55)
                    self.assertEqual(req.image.size, (75, 83))
                if mode.startswith("inpaint"):
                    self.assertEqual(kwargs["mask_image"].size, (72, 80))
                if mode != "img2img":
                    self.assertEqual((kwargs["width"], kwargs["height"]), (72, 80))

    def test_controlnet_uses_the_correct_image_parameter_in_each_mode(self):
        for mode in ("txt2img", "img2img", "inpaint_manual", "inpaint_auto"):
            with self.subTest(mode=mode):
                req = self.request(
                    mode,
                    controlnet_model="control",
                    controlnet_image=Image.new("L", (32, 32)),
                    controlnet_scale=0.7,
                    controlnet_start=0.1,
                    controlnet_end=0.9,
                )
                self.service._run_standard(req, 42)
                expected_mode = "inpaint" if mode.startswith("inpaint") else mode
                self.manager.get_controlnet.assert_called_with(expected_mode, "control")
                kwargs = self.pipe.call_args.kwargs
                control_key = "image" if mode == "txt2img" else "control_image"
                self.assertEqual(kwargs[control_key].size, (72, 80))
                self.assertEqual(kwargs[control_key].mode, "RGB")
                self.assertEqual(kwargs["controlnet_conditioning_scale"], 0.7)
                self.assertEqual(kwargs["control_guidance_start"], 0.1)
                self.assertEqual(kwargs["control_guidance_end"], 0.9)
                if mode == "txt2img":
                    self.assertNotIn("control_image", kwargs)

    def test_missing_control_image_still_fails_before_pipeline_execution(self):
        for mode in ("txt2img", "img2img", "inpaint_manual", "inpaint_auto"):
            with self.subTest(mode=mode):
                self.pipe.reset_mock()
                req = self.request(mode, controlnet_model="control")
                expected_mode = "inpaint" if mode.startswith("inpaint") else mode
                message = f"ControlNet {expected_mode} needs a control image"
                if mode != "txt2img":
                    message += " or Canny preprocess"
                with self.assertRaisesRegex(ValueError, message):
                    self.service._run_standard(req, 42)
                self.pipe.assert_not_called()

    def test_canny_uses_the_resized_source_image(self):
        for mode in ("img2img", "inpaint_manual"):
            with self.subTest(mode=mode):
                self.service._control_image = Mock(return_value=self.output)
                req = self.request(mode, controlnet_model="control", controlnet_preprocess="canny")
                self.service._run_standard(req, 42)
                args = self.service._control_image.call_args.args
                self.assertIs(args[0], req)
                self.assertEqual(args[1], (72, 80))
                self.assertEqual(args[2].size, (72, 80))

    def test_ip_adapter_image_is_converted_without_mutating_the_request(self):
        image = Image.new("RGBA", (32, 32))
        req = self.request(
            "txt2img",
            ip_adapter=SimpleNamespace(name="adapter"),
            ip_adapter_image=image,
            ip_adapter_scale=0.4,
        )
        self.service._run_standard(req, 42)
        self.assertEqual(self.pipe.call_args.kwargs["ip_adapter_image"].mode, "RGB")
        self.assertEqual(image.mode, "RGBA")
        self.manager.apply_ip_adapter.assert_called_once_with(self.pipe, req.ip_adapter, 0.4)

    def test_prepared_loras_are_not_applied_again(self):
        self.service._run_standard(self.request("txt2img", loras_prepared=True), 42)
        self.manager.apply_loras.assert_not_called()

    def test_missing_input_image_or_mask_still_fails(self):
        for mode in ("img2img", "inpaint_manual", "inpaint_auto"):
            with self.subTest(mode=mode):
                req = self.request(mode)
                req.image = None
                with self.assertRaises(ValueError):
                    self.service._run_standard(req, 42)
        req = self.request("inpaint_manual")
        req.mask = None
        with self.assertRaises(ValueError):
            self.service._run_standard(req, 42)


if __name__ == "__main__":
    unittest.main()
