# %% [markdown]
# # Looking Inside Gemma 4 While It Solves Perturbed Math Problems
#
# This is a companion to the Qwen3-4B tutorial notebook. It runs the same
# experiment with a different model, **Gemma 4 E4B** from Google DeepMind,
# so that we can check whether the layerwise patterns we saw are specific
# to one model or hold more generally.
#
# In this notebook, we will study a language model called **Gemma 4 E4B**. We will
# give it two related versions of a math problem:
#
# - a **Simple perturbation**
# - a **Hard perturbation**
#
# We will check whether the model answers each problem correctly. Then we will
# record the model's internal **hidden states** at every layer and measure how
# the Simple and Hard versions differ.
#
# ## Takeaways from this walkthrough
#
# By the end of the notebook, you should be able to explain:
#
# 1. What a token is.
# 2. What a transformer layer is.
# 3. What a hidden state is.
# 4. Why we compare matched Simple and Hard problems.
# 5. What relative L2 distance and cosine similarity measure.
# 6. The difference between an observational result and a causal result.
#
# You do **not** need to know calculus or linear algebra before starting.

# %% [markdown]
# ## 0. Smoke test: where does this notebook save files?
#
# Run this cell before doing anything else. It creates a tiny text file so that
# you can confirm which computer is running the notebook and where its files
# are stored.
#
# In Colab, the file will be created at:
#
# ```text
# /content/COLAB_SMOKE_TEST.txt
# ```
#
# This is on the temporary Colab computer, not on your Mac. Because you are
# connected to Colab through VS Code, the folder will not appear in your Mac's
# Finder or in VS Code's normal Explorer panel. The output below will still
# confirm that the remote file exists.

# %%
from pathlib import Path

runtime_root = (
    Path("/content")
    if Path("/content").is_dir()
    else Path.cwd()
)

smoke_test_path = runtime_root / "COLAB_SMOKE_TEST.txt"
smoke_test_path.write_text(
    "Success! This file was created by the notebook.\n"
    f"Runtime folder: {runtime_root}\n",
    encoding="utf-8",
)

print("Smoke-test file:", smoke_test_path)
print("Does it exist?", smoke_test_path.exists())
print("File contents:")
print(smoke_test_path.read_text(encoding="utf-8"))

print("Files and folders directly under", runtime_root)
for item in sorted(runtime_root.iterdir()):
    print(" -", item.name)

# %% [markdown]
# If the cell prints `Does it exist? True`, file saving is working. Later in
# the notebook, the real results will be placed inside the cloned repository:
#
# ```text
# /content/Exploring-memorization-in-hidden-states-of-LLMs-via-math-perturbations/
#     results/gemma4_math_perturb_tutorial/
# ```
#
# Like the smoke-test file, those results remain on the temporary Colab
# computer until you download them.
#
# ### How can a file exist without me being able to view the folder?
#
# The file is saved on a different computer.
#
# Your notebook appears in VS Code on your Mac, but its Python code runs on a
# temporary Google computer. That Google computer has its own folders,
# including:
#
# ```text
# /content
# ```
#
# When the notebook creates:
#
# ```text
# /content/COLAB_SMOKE_TEST.txt
# ```
#
# it saves the file inside Google's computer—not inside your Mac.
#
# VS Code sends code to that computer and receives the printed results.
# However, this connection does not add Google's folders to the VS Code
# Explorer. Therefore:
#
# - the file exists on Google's computer;
# - Python can read and change it;
# - VS Code can print its contents;
# - your Mac's Finder and VS Code Explorer cannot display its folder.
#
# It is similar to asking someone in another room to create a document on
# their computer. They can confirm that it exists and read it to you, but it
# does not appear in a folder on your computer. They must send the document to
# you before you can see it locally.

# %% [markdown]
# ## 0.1 Clone or update the GitHub repository
#
# The Colab computer also needs its own copy of the project repository.
#
# - **`git clone`** downloads the repository when the folder does not exist
#   yet. This normally happens once at the beginning of a new Colab session.
# - **`git pull`** checks GitHub for newer committed files when the repository
#   has already been cloned.
#
# The cell below decides which command is needed. It never deletes or
# overwrites the folder. `git pull --ff-only` also refuses to perform a
# complicated automatic merge.
#
# Remember that this creates or updates the copy on the temporary Colab
# computer. It does not change the repository folder on your Mac.

# %%
import subprocess

REPOSITORY_NAME = (
    "Exploring-memorization-in-hidden-states-of-LLMs-via-math-perturbations"
)
REPOSITORY_URL = (
    "https://github.com/yohaankharas906/"
    "Exploring-memorization-in-hidden-states-of-LLMs-via-math-perturbations.git"
)

current_folder = Path.cwd()

if (
    (current_folder / ".git").is_dir()
    and (current_folder / "MATH-Perturb").is_dir()
):
    REPO_ROOT = current_folder
else:
    REPO_ROOT = runtime_root / REPOSITORY_NAME

if (REPO_ROOT / ".git").is_dir():
    print("Repository already exists. Pulling committed updates from GitHub...")
    subprocess.check_call(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "pull",
            "--ff-only",
        ]
    )
elif REPO_ROOT.exists():
    raise FileExistsError(
        f"{REPO_ROOT} exists but is not a Git repository. "
        "Move or rename that folder before running this cell again."
    )
else:
    print("Repository is not present. Cloning it from GitHub...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            REPOSITORY_URL,
            str(REPO_ROOT),
        ]
    )

print("Remote repository folder:", REPO_ROOT)
print("Git folder found:", (REPO_ROOT / ".git").is_dir())

# %% [markdown]
# ## 1. The big idea
#
# Imagine that a math problem passes through a row of students. Each student
# reads the notes left by the previous student, adds something useful, and
# passes updated notes to the next student.
#
# A transformer model works somewhat like this:
#
# ```text
# tokens -> layer 1 -> layer 2 -> ... -> layer 42 -> answer
# ```
#
# At each layer, the model stores a long list of numbers. This list is called a
# **hidden state**. The numbers are not words that we can read directly, but we
# can compare two hidden states mathematically.
#
# In Gemma 4 E4B:
#
# - there are 42 transformer layers;
# - each hidden state contains 2,560 numbers (the same width as Qwen3-4B,
#   which makes the two notebooks easy to compare);
# - Hugging Face returns 43 hidden-state sites because it includes the initial
#   embedding representation and the final normalized representation.
#
# Our first question is:
#
# > As a Simple problem and its matched Hard problem move through the model,
# > when do their hidden states begin to differ?

# %% [markdown]
# ## 2. Start a GPU runtime
#
# In Google Colab, select:
#
# **Runtime -> Change runtime type -> T4 GPU**
#
# The following cell installs only the libraries that this notebook needs.
# Colab already provides PyTorch, so we do not reinstall it.

# %%
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers>=5.0,<6",  # Gemma 4 support requires a recent release
        "huggingface_hub>=0.24",
        "matplotlib>=3.8",
        "pandas>=2.0",
    ]
)

print("Setup complete.")

# %% [markdown]
# ## 3. Check the GPU
#
# A GPU performs the many matrix calculations used by a language model.
#
# Gemma 4 E4B stores about 8 billion parameters (including its large
# embeddings), which is roughly 16 GB in 16-bit precision. That does **not**
# fit on the free-tier T4 (about 15 GB). In Colab, choose:
#
# **Runtime -> Change runtime type -> L4 GPU** (or A100)
#
# If only a T4 is available, swap `MODEL_ID` in Section 4 to
# `google/gemma-4-E2B`, the smaller sibling model. Everything else in the
# notebook works unchanged, although the smaller model answers fewer problems
# correctly.

# %%
import gc
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU was found. In Colab, choose Runtime -> Change runtime type "
        "-> T4 GPU, then restart the notebook."
    )

DEVICE = torch.device("cuda")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)

# %% [markdown]
# ## 4. Experiment settings
#
# We use deterministic generation: the same prompt and settings should produce
# the same output. We also ask for only a boxed final answer. Short answers make
# behavioral screening faster and reduce the chance that generation ends
# before the model reaches its answer.
#
# `PROBLEM_ID = 9` is a short numerical problem that is convenient for the
# first demonstration.

# %%
MODEL_ID = "google/gemma-4-E4B"
# For a free-tier T4 GPU, use the smaller sibling instead:
# MODEL_ID = "google/gemma-4-E2B"

# Gemma models are trained in bfloat16; plain float16 can overflow and
# produce wrong or unstable hidden states, so we keep bfloat16 here.
MODEL_DTYPE = torch.bfloat16
ACTIVATION_DTYPE = np.float16

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512
PROBLEM_ID = 9

SYSTEM_PROMPT = (
    "Solve the mathematics problem. Respond with only the final answer "
    "inside \\boxed{}. Do not include reasoning or explanation."
)

RESULTS_FOLDER_NAME = "gemma4_math_perturb_tutorial"

# %% [markdown]
# ## 5. Load the paired dataset from this repository
#
# The Simple and Hard files are already stored in this project:
#
# ```text
# Exploring-memorization-in-hidden-states-of-LLMs-via-math-perturbations/
# └── MATH-Perturb/
#     └── math_perturb/
#         ├── math_perturb_simple.jsonl
#         └── math_perturb_hard.jsonl
# ```
#
# Each line is one JSON object, so the file type is called **JSONL**:
# JavaScript Object Notation Lines. Rows with the same `problem_id` come from
# the same original math problem.
#
# The code below first checks whether the current working directory is the
# repository. In Colab, it also checks the usual path made by cloning this
# repository into `/content`.
#
# Opening a notebook in Colab does **not** automatically copy the rest of its
# repository. If the data folder is not present, the code will clone the
# repository from GitHub. It will then read the Simple and Hard files from
# that local Colab copy.
#
# ### Why is the Git clone step necessary?
#
# VS Code and Google Colab may show the same notebook, but they are not running
# the code on the same computer.
#
# - The project you open in VS Code is stored on your Mac under a path such as
#   `/Users/your-name/Desktop/...`.
# - A Colab notebook runs on a temporary Google computer. Its files are stored
#   under `/content/...`.
#
# The Google computer cannot directly see folders on your Mac. VS Code is only
# displaying the notebook and connecting to the Colab runtime; it does not
# automatically send the complete project folder to Google.
#
# GitHub acts as the bridge:
#
# ```text
# project on your Mac -> GitHub -> temporary project copy in Colab
# ```
#
# The `git clone` command downloads the repository from GitHub onto the Colab
# computer. This gives the running notebook access to the MATH-Perturb data.
# The copy under `/content` is temporary and disappears when the Colab session
# ends. Files created there must be downloaded, or committed and pushed to
# GitHub, if you want to keep them.
#
# Important limitation:
#
# > This repository gives us the Simple and Hard versions, but not the original
# > clean prompt. Therefore, our activation difference is **Hard minus Simple**,
# > not perturbed minus clean.

# %%
possible_repo_roots = [
    Path.cwd(),
    Path.cwd() / REPOSITORY_NAME,
    Path("/content") / REPOSITORY_NAME,
]

def contains_dataset(root):
    """Return True when a folder contains the two MATH-Perturb data files."""

    data_folder = root / "MATH-Perturb" / "math_perturb"

    return (
        (data_folder / "math_perturb_simple.jsonl").is_file()
        and (data_folder / "math_perturb_hard.jsonl").is_file()
    )


REPO_ROOT = next(
    (root for root in possible_repo_roots if contains_dataset(root)),
    None,
)

if REPO_ROOT is None:
    clone_destination = Path("/content") / REPOSITORY_NAME

    if clone_destination.exists():
        raise FileNotFoundError(
            f"{clone_destination} exists, but it does not contain both "
            "MATH-Perturb data files. Rename or remove that incomplete folder, "
            "then run this cell again."
        )

    print("The dataset is not present in this Colab session.")
    print("Cloning the project repository now...")

    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            REPOSITORY_URL,
            str(clone_destination),
        ]
    )

    REPO_ROOT = clone_destination

if not contains_dataset(REPO_ROOT):
    raise FileNotFoundError(
        "The repository was found, but the Simple and Hard data files are "
        "missing."
    )

DATA_DIR = REPO_ROOT / "MATH-Perturb" / "math_perturb"
SIMPLE_PATH = DATA_DIR / "math_perturb_simple.jsonl"
HARD_PATH = DATA_DIR / "math_perturb_hard.jsonl"

# Save generated files inside the same local repository as the dataset.
OUTPUT_DIR = REPO_ROOT / "results" / RESULTS_FOLDER_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    """Read a JSONL file and return a list of Python dictionaries."""

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def index_by_problem_id(rows):
    """Create a lookup table: problem_id -> dataset row."""

    indexed = {}

    for row in rows:
        problem_id = row["problem_id"]

        if problem_id in indexed:
            raise ValueError(f"Duplicate problem_id: {problem_id}")

        indexed[problem_id] = row

    return indexed


simple_rows = index_by_problem_id(load_jsonl(SIMPLE_PATH))
hard_rows = index_by_problem_id(load_jsonl(HARD_PATH))
paired_ids = sorted(set(simple_rows) & set(hard_rows))

print("Repository:", REPO_ROOT)
print("Data folder:", DATA_DIR)
print("Results folder:", OUTPUT_DIR)
print("Simple problems:", len(simple_rows))
print("Hard problems:", len(hard_rows))
print("Matched pairs:", len(paired_ids))

# %% [markdown]
# ## 6. Look at one matched pair
#
# Notice that the Simple and Hard problems may have different correct answers.
# We are not checking whether their text is identical. We are checking how the
# model processes two related modifications of the same source problem.

# %%
simple_example = simple_rows[PROBLEM_ID]
hard_example = hard_rows[PROBLEM_ID]

print("SIMPLE PROBLEM")
print(simple_example["problem"])
print("Correct answer:", simple_example["answer"])
print()
print("HARD PROBLEM")
print(hard_example["problem"])
print("Correct answer:", hard_example["answer"])

# %% [markdown]
# ## 7. Load Gemma 4 E4B
#
# A parameter is a number learned during model training. "E4B" means that this
# model behaves like a four-billion-parameter model during inference (its
# **effective** size), although it stores about eight billion numbers in total
# once its large vocabulary and per-layer embeddings are counted.
#
# We use `bfloat16`, meaning that each model value is stored with 16 bits
# instead of the usual 32. This reduces GPU memory use. We do **not** quantize
# the model, because we want its hidden states to remain straightforward to
# interpret.
#
# Gemma 4 is a multimodal model (it can also accept images and audio), so its
# configuration keeps the text-model details in a `text_config` sub-object.
# We only use the text side in this notebook.

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.truncation_side = "left"

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=MODEL_DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(DEVICE)
except ValueError:
    # Gemma 4 repositories register a multimodal wrapper class. Text-only
    # prompts still work: we simply load that wrapper instead.
    from transformers import AutoModelForMultimodalLM

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=MODEL_DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(DEVICE)

model.eval()

# The text-model details live in text_config for multimodal Gemma checkpoints.
text_config = getattr(model.config, "text_config", model.config)

print("Transformer layers:", text_config.num_hidden_layers)
print("Hidden-state width:", text_config.hidden_size)
print(
    "GPU memory currently allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)

# %% [markdown]
# ## 8. Tokens and prompts
#
# A language model does not read complete words directly. Its tokenizer breaks
# text into **tokens**, which may be words, pieces of words, punctuation, or
# special markers.
#
# A chat template adds markers that tell the model which text belongs to the
# user and where the assistant should begin answering.
#
# We disable Gemma's visible thinking mode so that responses are shorter and
# easier to score.

# %%
def make_prompt(problem):
    """Turn a math problem into Gemma's formatted chat prompt."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


example_prompt = make_prompt(simple_example["problem"])
example_tokens = tokenizer(example_prompt)["input_ids"]

print("Number of tokens in the formatted Simple prompt:", len(example_tokens))
print()
print("Beginning of formatted prompt:")
print(example_prompt[:500])

# %% [markdown]
# ## 9. Which hidden states will we save?
#
# At every hidden-state site, Gemma 4 E4B gives **every token** a list of 2,560
# numbers. Think of that list as the model's private set of notes about that
# token. The notes begin as a simple token lookup, but after each transformer
# layer they can contain information gathered from other tokens in the prompt.
#
# For example, suppose a formatted prompt contains 500 tokens. Saving the full
# hidden states would require:
#
# ```text
# 43 sites x 500 tokens x 2,560 numbers
# ```
#
# That is about 55 million numbers for just one prompt. In 16-bit format,
# it is about 110 MB. Two versions of hundreds of problems would quickly use
# many gigabytes.
#
# Instead of keeping every token vector, we turn all the token vectors at each
# site into three smaller summaries. Each summary still contains 2,560
# numbers, but we save only one summary vector per site:
#
# 1. **last_token**: the representation at the final prompt token.
#    The final token is located exactly where the model is about to begin its
#    answer. Although it is only one token, its deeper-layer representation
#    can contain information gathered from the whole problem. Imagine asking
#    a student to write one final set of notes immediately before answering.
#
# 2. **last_three_mean**: the average of the last three prompt-token
#    representations. To average vectors, we average the first number across
#    the three vectors, then the second number, and so on through all 2,560
#    positions. Using three tokens makes the measurement less dependent on one
#    possibly unusual final token. This choice is inspired by the CPE paper.
#
# 3. **prompt_mean**: the average representation across the complete prompt.
#    This gives a broad summary of the whole input. It may capture widespread
#    changes, but it can also blur a small and important change because all
#    tokens are mixed together.
#
# None of these vectors is automatically “the model's answer” or “the model's
# reasoning.” They are compact measurements that let us ask where the Simple
# and Hard representations become more or less similar.
#
# ### Why the final token gives us a built-in check
#
# Think of **hidden-state site 0** as the moment before the model has started
# reading information across the prompt. The model has turned each token into
# its starting list of numbers, called an **embedding**, but the transformer
# layers have not yet updated those numbers using the surrounding words.
#
# An analogy is a student receiving two worksheets. Both worksheets end with
# the same blank labeled “Answer:”. Before the student reads either question,
# the “Answer:” label is the same on both pages. After the student reads the
# different questions, the notes they place beside “Answer:” may become
# different.
#
# Our formatted Simple and Hard prompts work in a similar way. They contain
# different math problems, but both end with the same special assistant marker
# that tells Gemma to begin answering. At site 0, the final marker has not yet
# gathered information from the math problem. Its starting representation
# should therefore be the same in both prompts.
#
# This gives us the following expectations:
#
# | Summary | What should happen at site 0? | Why? |
# |---|---|---|
# | `last_token` | The Simple and Hard values should match, or be extremely close. | The final token is the same in both prompts. |
# | `last_three_mean` | The values should match if the final three tokens are the same. | We are averaging the same three starting token representations. |
# | `prompt_mean` | The values can already be different. | This average includes all the different words and numbers in the two math problems. |
#
# This is a **sanity check** for our code. If `last_token` is very different at
# site 0, the difference is probably not an interesting model result. We should
# first check whether the prompts really end with the same token and whether
# tokenization, padding, or pooling was handled correctly.
#
# After site 0, each transformer layer lets the final token gather information
# from earlier tokens. The Simple and Hard `last_token` representations may
# then move apart because the earlier math problems are different. Seeing this
# change tells us that context is affecting the final-token representation.
# It does **not** yet prove that the observed difference caused the model to
# produce a particular answer.
#
# ### What the next code cell does
#
# The function below:
#
# 1. counts the real prompt tokens and ignores any padding;
# 2. selects the last token, last three tokens, or all prompt tokens;
# 3. averages when a summary uses more than one token;
# 4. repeats this for every hidden-state site;
# 5. moves the small summaries to CPU memory and stores them in 16-bit format.
#
# The output shape for each summary is `(43, 2560)`: one row for each
# hidden-state site and 2,560 measurements in each row.

# %%
SUMMARY_NAMES = (
    "last_token",
    "last_three_mean",
    "prompt_mean",
)


def pool_hidden_states(hidden_states, attention_mask):
    """Convert full hidden states into three small layer-by-layer summaries."""

    valid_tokens = int(attention_mask[0].sum().item())
    last_three_start = max(0, valid_tokens - 3)

    pooled_tensors = {
        "last_token": [
            state[0, valid_tokens - 1]
            for state in hidden_states
        ],
        "last_three_mean": [
            state[0, last_three_start:valid_tokens].mean(dim=0)
            for state in hidden_states
        ],
        "prompt_mean": [
            state[0, :valid_tokens].mean(dim=0)
            for state in hidden_states
        ],
    }

    pooled_arrays = {}

    for name, values in pooled_tensors.items():
        pooled_arrays[name] = np.stack(
            [
                value.detach().float().cpu().numpy()
                for value in values
            ]
        ).astype(ACTIVATION_DTYPE)

    return pooled_arrays

# %% [markdown]
# ## 10. Create functions for extracting and grading answers
#
# Our experiment has two connected parts:
#
# 1. **Behavior:** Did Gemma answer the Simple and Hard problems correctly?
# 2. **Hidden states:** What happened inside Gemma while it processed them?
#
# Section 9 prepared the hidden-state measurements. This section prepares the
# behavioral check. It creates helper functions that we will call later, after
# Gemma generates an answer.
#
# ```text
# Gemma generates a response
#             ↓
# extract the answer inside \boxed{...}
#             ↓
# compare it with the dataset's correct answer
#             ↓
# label the response correct or wrong
# ```
#
# Section 10 does **not** run Gemma or collect hidden states. It only defines
# the small “answer grader” functions that later sections will use. We need
# these labels so that we can eventually compare cases such as:
#
# - `C->C`: Gemma answers both Simple and Hard correctly;
# - `C->W`: Gemma answers Simple correctly but Hard incorrectly.
#
# Without this step, we could measure differences between hidden states, but
# we would not know whether those differences occurred during success or
# failure.
#
# The function below finds the final `\boxed{...}` answer. It counts nested
# braces, so it can handle answers such as `\boxed{\frac{1}{2}}`.
#
# > **Side note: Why do we start with numerical answers?**
# >
# > For this first test, we use problems whose answers are ordinary numbers
# > because they are easy to grade automatically. For example, we can directly
# > check whether the model's answer `69` matches the correct answer `69`.
# >
# > Algebraic answers are harder because two expressions can look different
# > but still be mathematically equal. For example, `2(x+1)` and `2x+2` have
# > different text but the same mathematical value. A basic text checker might
# > incorrectly mark one of them as wrong. A larger experiment should therefore
# > use MATH-Perturb's specialized answer checker, which is designed to
# > recognize mathematically equivalent answers.

# %%
def extract_boxed_answer(text):
    """Return the contents of the final boxed answer, or None."""

    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    position = start + len(marker)
    depth = 1
    characters = []

    while position < len(text):
        character = text[position]

        if character == "{":
            depth += 1
            characters.append(character)
        elif character == "}":
            depth -= 1

            if depth == 0:
                return "".join(characters)

            characters.append(character)
        else:
            characters.append(character)

        position += 1

    return None


def normalize_answer(answer):
    """Remove spaces and display-math markers for a basic text comparison."""

    if answer is None:
        return None

    return re.sub(r"\s+", "", str(answer)).strip("$")


def compare_numeric_answer(response, ground_truth):
    """Compare a boxed prediction with a numerical ground truth."""

    extracted = extract_boxed_answer(response)

    if extracted is None:
        return None, False

    try:
        prediction_number = float(extracted.replace(",", ""))
        ground_truth_number = float(str(ground_truth).replace(",", ""))

        return extracted, abs(prediction_number - ground_truth_number) < 1e-8
    except ValueError:
        return (
            extracted,
            normalize_answer(extracted) == normalize_answer(ground_truth),
        )

# %% [markdown]
# ## 11. Run the model and collect prompt-boundary activations
#
# Section 11 creates a function that handles one math problem.
#
# It gives the problem to Gemma and records the model's hidden states
# immediately after Gemma reads the problem, but before it starts writing an
# answer. It then asks Gemma to generate its answer, extracts the final answer,
# and checks whether it matches the correct answer from the dataset.
#
# The function returns two things:
#
# - the model's behavior: its response, extracted answer, and whether it was
#   correct;
# - the model's internal information: the three hidden-state summaries from
#   every layer.
#
# Section 11 only creates this function. Section 12 uses it on the first Simple
# and Hard problem pair.
#
# The saved activations therefore describe the model at the **end of the input
# prompt, before any answer tokens are generated**. They do not record every
# hidden state created while Gemma writes its answer.

# %%
def run_variant(row, max_new_tokens=MAX_NEW_TOKENS):
    """Collect prompt activations and generate an answer for one dataset row."""

    prompt = make_prompt(row["problem"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    activations = pool_hidden_states(
        outputs.hidden_states,
        inputs["attention_mask"],
    )

    number_of_input_tokens = int(
        inputs["attention_mask"][0].sum().item()
    )

    del outputs
    torch.cuda.empty_cache()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    extracted, is_correct = compare_numeric_answer(
        response,
        row["answer"],
    )

    result = {
        "problem": row["problem"],
        "ground_truth": row["answer"],
        "response": response,
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "input_tokens": number_of_input_tokens,
        "generated_tokens": int(new_tokens.shape[0]),
    }

    del inputs, generated, new_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return result, activations

# %% [markdown]
# ## 12. Run the first Simple/Hard pair
#
# Section 12 chooses one matched pair of math problems: one Simple version and
# one Hard version.
#
# It gives the Simple problem to Gemma, records the model's internal hidden
# states, and checks its answer. It then repeats the same process with the Hard
# version.
#
# After both runs, it displays the correct answers, Gemma's answers, and whether
# Gemma was correct. It also checks that the hidden states were saved in the
# expected format.
#
# This is a small practice run. Its purpose is to make sure the entire
# experiment works correctly on one pair before testing many problems.
#
# This cell may take a little while the first time. The two activation
# dictionaries should each contain arrays with shape `(43, 2560)`.

# %%
simple_result, simple_activations = run_variant(simple_example)
hard_result, hard_activations = run_variant(hard_example)

print("SIMPLE")
print("Ground truth:", simple_result["ground_truth"])
print("Prediction:", simple_result["extracted_answer"])
print("Correct:", simple_result["is_correct"])
print("Response:", simple_result["response"])
print()
print("HARD")
print("Ground truth:", hard_result["ground_truth"])
print("Prediction:", hard_result["extracted_answer"])
print("Correct:", hard_result["is_correct"])
print("Response:", hard_result["response"])
print()

for summary_name in SUMMARY_NAMES:
    print(
        summary_name,
        simple_activations[summary_name].shape,
        simple_activations[summary_name].dtype,
    )

# %% [markdown]
# ## 13. Describe the behavioral transition
#
# We use four labels:
#
# - **C->C**: Simple correct, Hard correct.
# - **C->W**: Simple correct, Hard wrong.
# - **W->C**: Simple wrong, Hard correct.
# - **W->W**: both wrong.
#
# `C->W` pairs are especially useful when studying loss of robustness. `C->C`
# pairs provide a successful baseline.

# %%
def transition_label(simple_correct, hard_correct):
    """Create a short correctness-transition label."""

    return (
        f"{'C' if simple_correct else 'W'}->"
        f"{'C' if hard_correct else 'W'}"
    )


first_transition = transition_label(
    simple_result["is_correct"],
    hard_result["is_correct"],
)

print("Behavioral transition:", first_transition)

# %% [markdown]
# ## 14. Measure hidden-state differences
#
# ### Relative L2 distance
#
# L2 distance is the ordinary straight-line distance between two lists of
# numbers. We divide it by the typical size of the two hidden states so that
# values from different layers are easier to compare.
#
# A relative L2 value of `0.04` means that the difference is about 4% of the
# typical hidden-state magnitude.
#
# ### Cosine similarity
#
# Cosine similarity compares direction:
#
# - `1.0`: almost exactly the same direction.
# - `0.0`: unrelated/sideways directions.
# - `-1.0`: opposite directions.
#
# Two vectors can have high cosine similarity while still having a meaningful
# L2 difference. Cosine ignores much of the magnitude information.

# %%
def layerwise_metrics(simple, hard):
    """Calculate one distance and one similarity value at every site."""

    simple = simple.astype(np.float32)
    hard = hard.astype(np.float32)
    difference = hard - simple

    l2_shift = np.linalg.norm(difference, axis=1)

    baseline_norm = 0.5 * (
        np.linalg.norm(simple, axis=1)
        + np.linalg.norm(hard, axis=1)
    )

    relative_l2 = l2_shift / np.maximum(
        baseline_norm,
        1e-12,
    )

    cosine = np.sum(simple * hard, axis=1) / np.maximum(
        np.linalg.norm(simple, axis=1)
        * np.linalg.norm(hard, axis=1),
        1e-12,
    )

    return {
        "l2": l2_shift,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


first_metrics = {
    summary_name: layerwise_metrics(
        simple_activations[summary_name],
        hard_activations[summary_name],
    )
    for summary_name in SUMMARY_NAMES
}

print("Metrics created:", list(first_metrics))

# %% [markdown]
# ## 15. Plot the layerwise pattern
#
# Expected sanity check:
#
# - `last_token` and `last_three_mean` should be zero or nearly zero at site 0,
#   because the final chat-template tokens are identical.
# - Differences can then grow through later layers as the model combines those
#   shared tokens with different problem contexts.
# - `prompt_mean` may already differ at site 0 because the two problems contain
#   different tokens and may have different lengths.

# %%
def plot_pair_metrics(metrics, problem_id, transition):
    """Plot relative L2 distance and cosine similarity for one pair."""

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    for summary_name, summary_metrics in metrics.items():
        sites = np.arange(len(summary_metrics["relative_l2"]))

        axes[0].plot(
            sites,
            summary_metrics["relative_l2"],
            marker="o",
            markersize=3,
            label=summary_name,
        )
        axes[1].plot(
            sites,
            summary_metrics["cosine"],
            marker="o",
            markersize=3,
            label=summary_name,
        )

    axes[0].set_title(
        f"Gemma 4 E4B Simple to Hard shift: problem {problem_id} ({transition})"
    )
    axes[0].set_ylabel("Relative L2 shift")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel(
        "Hidden-state site (0 = embedding; final site = final norm)"
    )
    axes[1].set_ylabel("Cosine similarity")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    return figure


first_figure = plot_pair_metrics(
    first_metrics,
    PROBLEM_ID,
    first_transition,
)

first_plot_path = OUTPUT_DIR / f"problem_{PROBLEM_ID}_layerwise.png"
first_figure.savefig(first_plot_path, dpi=180)
plt.show()

print("Saved:", first_plot_path)

# %% [markdown]
# ## 16. Print selected sites
#
# A table makes the plot easier to discuss in a report. Site 42 is the final
# normalized representation for this model, not a 43rd transformer block.

# %%
selected_sites = [0, 1, 8, 14, 21, 28, 35, 42]

for summary_name, values in first_metrics.items():
    print(f"\n{summary_name}")
    print("site | relative L2 | cosine")

    for site in selected_sites:
        print(
            f"{site:>4} | "
            f"{values['relative_l2'][site]:>11.5f} | "
            f"{values['cosine'][site]:>7.5f}"
        )

# %% [markdown]
# ## 17. Screen several pairs behaviorally
#
# One example can show that the code works, but it cannot establish a scientific
# pattern. We next generate answers for several numerical-answer pairs without
# collecting activations. This is cheaper than saving activations for every
# example.
#
# We first look for:
#
# - a few `C->C` successful pairs;
# - a few `C->W` possible robustness failures.
#
# A missing boxed answer is labeled `NO_BOX`, not wrong. We should never confuse
# truncated or badly formatted output with a reasoning failure.

# %%
def is_numeric_answer(answer):
    """Return True when an answer can be read directly as a number."""

    try:
        float(str(answer).replace(",", ""))
        return True
    except ValueError:
        return False


def generate_only(row, max_new_tokens=MAX_NEW_TOKENS):
    """Generate an answer without saving hidden states."""

    prompt = make_prompt(row["problem"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    del inputs, generated, new_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return response


def screen_problem_ids(problem_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Generate Simple and Hard answers and classify each pair."""

    results = []

    for problem_id in tqdm(problem_ids, desc="Screening pairs"):
        simple_row = simple_rows[problem_id]
        hard_row = hard_rows[problem_id]

        simple_response = generate_only(
            simple_row,
            max_new_tokens=max_new_tokens,
        )
        hard_response = generate_only(
            hard_row,
            max_new_tokens=max_new_tokens,
        )

        simple_prediction, simple_correct = compare_numeric_answer(
            simple_response,
            simple_row["answer"],
        )
        hard_prediction, hard_correct = compare_numeric_answer(
            hard_response,
            hard_row["answer"],
        )

        if simple_prediction is None or hard_prediction is None:
            transition = "NO_BOX"
        else:
            transition = transition_label(
                simple_correct,
                hard_correct,
            )

        results.append(
            {
                "problem_id": problem_id,
                "type": simple_row.get("type"),
                "simple_ground_truth": simple_row["answer"],
                "simple_prediction": simple_prediction,
                "simple_correct": simple_correct,
                "hard_ground_truth": hard_row["answer"],
                "hard_prediction": hard_prediction,
                "hard_correct": hard_correct,
                "transition": transition,
                "simple_response": simple_response,
                "hard_response": hard_response,
            }
        )

    return results


numeric_pair_ids = [
    problem_id
    for problem_id in paired_ids
    if problem_id != PROBLEM_ID
    and is_numeric_answer(simple_rows[problem_id]["answer"])
    and is_numeric_answer(hard_rows[problem_id]["answer"])
]

NUMBER_TO_SCREEN = 10
screen_ids = numeric_pair_ids[:NUMBER_TO_SCREEN]

print("Problem IDs to screen:", screen_ids)

# %% [markdown]
# The next cell runs 20 generations: one Simple and one Hard prompt for each of
# 10 pairs. It can take several minutes on a T4.

# %%
screen_results = screen_problem_ids(
    screen_ids,
    max_new_tokens=MAX_NEW_TOKENS,
)

summary_columns = [
    "problem_id",
    "type",
    "simple_ground_truth",
    "simple_prediction",
    "simple_correct",
    "hard_ground_truth",
    "hard_prediction",
    "hard_correct",
    "transition",
]

screen_table = pd.DataFrame(screen_results)[summary_columns]
display(screen_table)

print("\nTransition counts:")
print(screen_table["transition"].value_counts())

# %% [markdown]
# ## 18. Inspect candidate failures
#
# Automated labels are only a first filter. Before calling something a model
# failure, read the problem, prediction, and ground truth. This is especially
# important for symbolic answers, units, multiple answers, and equivalent
# fractions.

# %%
failure_candidates = [
    result
    for result in screen_results
    if result["transition"] == "C->W"
]

print("C->W candidates:", len(failure_candidates))

for result in failure_candidates:
    problem_id = result["problem_id"]

    print("\n" + "=" * 80)
    print("Problem ID:", problem_id)

    print("\nSIMPLE PROBLEM")
    print(simple_rows[problem_id]["problem"])
    print("Prediction:", result["simple_prediction"])
    print("Ground truth:", result["simple_ground_truth"])

    print("\nHARD PROBLEM")
    print(hard_rows[problem_id]["problem"])
    print("Prediction:", result["hard_prediction"])
    print("Ground truth:", result["hard_ground_truth"])
    print("Raw response:", result["hard_response"])

# %% [markdown]
# ## 19. Optional: compare several successful and failed pairs
#
# Run this section only after the screening table contains manually confirmed
# `C->C` and `C->W` examples.
#
# We collect prompt activations for up to three examples from each group. Then
# we average their layerwise curves. With only a few examples, the result is
# exploratory; a final study should use more examples and confidence intervals.

# %%
def collect_prompt_activations(row):
    """Collect pooled hidden states without generating an answer."""

    prompt = make_prompt(row["problem"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    activations = pool_hidden_states(
        outputs.hidden_states,
        inputs["attention_mask"],
    )

    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return activations


def collect_pair_metrics(problem_id):
    """Collect Simple/Hard prompt activations and calculate their metrics."""

    simple_values = collect_prompt_activations(
        simple_rows[problem_id]
    )
    hard_values = collect_prompt_activations(
        hard_rows[problem_id]
    )

    return {
        summary_name: layerwise_metrics(
            simple_values[summary_name],
            hard_values[summary_name],
        )
        for summary_name in SUMMARY_NAMES
    }


c_to_c_ids = [
    result["problem_id"]
    for result in screen_results
    if result["transition"] == "C->C"
][:3]

c_to_w_ids = [
    result["problem_id"]
    for result in screen_results
    if result["transition"] == "C->W"
][:3]

print("C->C IDs selected:", c_to_c_ids)
print("C->W IDs selected:", c_to_w_ids)

# %%
group_pair_metrics = {
    "C->C": {},
    "C->W": {},
}

for group_name, problem_ids in (
    ("C->C", c_to_c_ids),
    ("C->W", c_to_w_ids),
):
    for problem_id in tqdm(
        problem_ids,
        desc=f"Collecting {group_name}",
    ):
        group_pair_metrics[group_name][problem_id] = (
            collect_pair_metrics(problem_id)
        )

print("Activation collection complete.")

# %% [markdown]
# ## 20. Plot group-average curves
#
# The plot below uses the final-token representation because it has the
# cleanest site-0 sanity check.
#
# Possible hypothesis:
#
# > Failed `C->W` pairs may show an earlier, larger, or differently shaped
# > Simple-to-Hard shift than successful `C->C` pairs.
#
# This is a hypothesis, not something we should assume in advance.

# %%
def group_metric_matrix(group_name, summary_name, metric_name):
    """Stack one layerwise metric from every pair in a group."""

    rows = []

    for pair_metrics in group_pair_metrics[group_name].values():
        rows.append(pair_metrics[summary_name][metric_name])

    if not rows:
        return None

    return np.stack(rows)


figure, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True,
)

for group_name, color in (
    ("C->C", "tab:blue"),
    ("C->W", "tab:red"),
):
    relative_l2_matrix = group_metric_matrix(
        group_name,
        "last_token",
        "relative_l2",
    )
    cosine_matrix = group_metric_matrix(
        group_name,
        "last_token",
        "cosine",
    )

    if relative_l2_matrix is None:
        print(f"No examples available for {group_name}.")
        continue

    sites = np.arange(relative_l2_matrix.shape[1])

    axes[0].plot(
        sites,
        relative_l2_matrix.mean(axis=0),
        label=f"{group_name} (n={len(relative_l2_matrix)})",
        color=color,
    )
    axes[1].plot(
        sites,
        cosine_matrix.mean(axis=0),
        label=f"{group_name} (n={len(cosine_matrix)})",
        color=color,
    )

    if len(relative_l2_matrix) > 1:
        relative_standard_error = (
            relative_l2_matrix.std(axis=0, ddof=1)
            / np.sqrt(len(relative_l2_matrix))
        )
        cosine_standard_error = (
            cosine_matrix.std(axis=0, ddof=1)
            / np.sqrt(len(cosine_matrix))
        )

        axes[0].fill_between(
            sites,
            relative_l2_matrix.mean(axis=0) - relative_standard_error,
            relative_l2_matrix.mean(axis=0) + relative_standard_error,
            color=color,
            alpha=0.2,
        )
        axes[1].fill_between(
            sites,
            cosine_matrix.mean(axis=0) - cosine_standard_error,
            cosine_matrix.mean(axis=0) + cosine_standard_error,
            color=color,
            alpha=0.2,
        )

axes[0].set_title("Successful versus failed Simple-to-Hard shifts")
axes[0].set_ylabel("Mean relative L2 shift")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].set_xlabel("Hidden-state site")
axes[1].set_ylabel("Mean cosine similarity")
axes[1].grid(alpha=0.25)
axes[1].legend()

figure.tight_layout()

group_plot_path = OUTPUT_DIR / "group_comparison.png"
figure.savefig(group_plot_path, dpi=180)
plt.show()

print("Saved:", group_plot_path)

# %% [markdown]
# ## 21. Save the results
#
# The individual result files are saved locally inside:
#
# ```text
# results/gemma4_math_perturb_tutorial/
# ```
#
# Here, “locally” means the computer running the notebook. If you run the
# notebook on your own computer, the files stay inside your repository. If you
# run it in Colab, they are inside Colab's temporary copy of the repository.
# Colab deletes that temporary computer when the session ends, so download the
# ZIP file before closing the notebook.

# %%
first_activation_path = OUTPUT_DIR / f"problem_{PROBLEM_ID}_activations.npz"

np.savez_compressed(
    first_activation_path,
    **{
        f"simple_{summary_name}": simple_activations[summary_name]
        for summary_name in SUMMARY_NAMES
    },
    **{
        f"hard_{summary_name}": hard_activations[summary_name]
        for summary_name in SUMMARY_NAMES
    },
)

screening_path = OUTPUT_DIR / "screening_results.json"
screening_path.write_text(
    json.dumps(screen_results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Saved:", first_activation_path)
print("Saved:", screening_path)

# %%
import shutil

archive_base = OUTPUT_DIR.parent / "gemma4_math_perturb_tutorial_results"

archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    OUTPUT_DIR,
)

print("Created:", archive_path)

# %% [markdown]
# In Colab, the following cell downloads the archive to your own computer. If
# you are running locally, the archive is already in the repository's
# `results` folder.

# %%
try:
    from google.colab import files

    files.download(archive_path)
except ImportError:
    print("Running outside Colab. Your ZIP file is already here:")
    print(archive_path)

# %% [markdown]
# ## 22. What can we conclude?
#
# ### A valid conclusion from one pair
#
# > The measurement pipeline detects how a matched Simple/Hard difference
# > develops across Gemma 4 E4B's hidden-state sites.
#
# ### Conclusions we cannot make from one pair
#
# We cannot yet say:
#
# - that Hard perturbations always cause larger changes;
# - that a specific layer causes failure;
# - that the model memorized a problem;
# - that the pattern generalizes to other math categories.
#
# ### Observational versus causal evidence
#
# Everything in this notebook is **observational**. We measure internal
# differences and relate them to behavior.
#
# A stronger mechanistic experiment would perform **activation patching**:
#
# 1. Run a Simple problem and save an activation.
# 2. Run its failed Hard partner.
# 3. Replace the Hard activation at one layer with the Simple activation.
# 4. Check whether the correct Hard answer becomes more likely.
#
# If a carefully controlled patch reliably changes behavior on held-out
# examples, that is causal evidence. Activation patching should be the next
# notebook after this baseline is working.

# %% [markdown]
# ## 23. Write your first draft in Overleaf
#
# After producing the plot, begin the first report draft in Overleaf. Explain
# the research question, why Simple and Hard perturbations are being compared,
# why Gemma 4 E4B was selected, how the matched data were loaded, how
# correctness
# was checked, and how hidden states were summarized.
#
# Describe the three summaries—`last_token`, `last_three_mean`, and
# `prompt_mean`—and explain relative L2 distance and cosine similarity in your
# own words.
#
# In the preliminary-results section, state that this is a one-pair smoke test
# rather than a final experiment. Report whether the pair was `C->C`, `C->W`,
# `W->C`, or `W->W`. Describe what the plot shows, including where the Simple
# and Hard representations begin separating most strongly. Include the plot as
# a preliminary figure.
#
# Finish the draft by explaining the limitations. One pair cannot establish a
# general pattern, prove memorization, or show that a particular layer caused
# the answer. State that the next milestone will be to screen more matched
# pairs and compare the average layerwise patterns of `C->C` and `C->W`
# examples.
