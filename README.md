# Direct Preference Optimization

Training Large Language Models (LLMs) on extensive datasets in an unsupervised manner has proven highly effective in creating models capable of a wide range of tasks. These models demonstrate a significant breadth of knowledge and understanding of the world. For most applications, it’s crucial for LLMs to generate text that is contextually consistent and aligned with the intended task and user behavior. This includes developing LLMs that are safe, aligned, and unbiased, or those capable of generating syntactically and functionally correct code, despite the presence of incorrect code in the training data. However, the pre-training process alone does not guarantee specific model behavior. This is where Reinforcement Learning From Human Feedback (RLHF) becomes vital.

RLHF is a technique used to fine-tune LLMs by maximizing a reward function derived from another reward model trained on human feedback from evaluators based on a set of generated samples. This technique is widely used and is considered state-of-the-art. However, RLHF has several drawbacks that limit its effectiveness as a solution.

Direct Preference Optimization (DPO), a newly proposed technique addresses these drawbacks and offers a more robust solution. In this project, we delve into the concept of Direct Preference Optimization (DPO) as introduced in the award-winning paper at NeurIPS 2023. We will explore the process of RLHF, its limitations, and how DPO effectively overcomes these challenges. Additionally, I will provide and explain practical guides both on coding DPO from scratch in PyTorch as well as using the HuggingFace DPOTrainer API.



