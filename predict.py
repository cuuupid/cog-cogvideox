from cog import BasePredictor, Path, Input
import subprocess
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
# from pget import pget_manifest

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

class Predictor(BasePredictor):

    def setup(self):
        # pget_manifest()
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained("./weights/glm-4", trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained("./weights/glm-4",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.pipe = CogVideoXPipeline.from_pretrained("./weights/cog-video-x", torch_dtype=torch.bfloat16)
        # for 2b, we should switch this to CogVideoXDDIMScheduler
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()

    def predict(self,
        prompt: str = Input(description="Prompt"),
        extend_prompt: bool = Input(description="If enabled, will use GLM-4 to make the prompt long (as intended for CogVideoX).", default=True),
        steps: int = Input(description="# of inference steps, more steps can improve quality.", default=50, ge=1, le=120),
        guidance: float = Input(description="The scale for classifier-free guidance, higher guidance can improve adherence to your prompt.", default=6.0, ge=0, le=12.0),
        num_outputs: int = Input(description="# of output videos", default=1),
        seed: int = Input(description="Seed for reproducibility.", default=42)
    ) -> Path:
        print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
        if extend_prompt:
            text = prompt.strip()
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": f"{sys_prompt}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
                    },
                    {
                        "role": "assistant",
                        "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
                    },
                    {
                        "role": "assistant",
                        "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                    },
                    {
                        "role": "assistant",
                        "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                    },
                    {
                        "role": "user",
                        "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
                    },
                ],
                add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
            ).to(device)
            gen_kwargs = { "max_length": 250, "do_sample": True, "top_k": 0.7, "temperature": 0.01,  }
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("Extended prompt:", prompt, sep='\n')
        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=num_outputs,
            num_frames=49, # avoid tweaking this
            use_dynamic_cfg=True,
            guidance_scale=guidance,
            generator=torch.Generator().manual_seed(seed)
        ).frames[0]
        print("Exporting to MP4...")
        export_to_video(video, "output.mp4", fps=8) # avoid tweaking this
        return Path("output.mp4")
