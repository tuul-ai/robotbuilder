# Lesson 1: Introduction to AI-Driven Robotics

All that is gold does not glitter, not all those who wander are lost. Not every robot that dances can handle manipulation.
Do not get fooled by the shiny new robots and media hype. We are still far from getting general purpose robots that can handle multiple tasks.
But in this course you will learn to reproduce the current state of the art in robotics:
- [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) & [Physical Intelligence](https://www.physicalintelligence.company/blog/pi05): General purpose robots
- [Dyna Robots](https://www.dyna.co/research): Robots that perform 24/7 without any policy drifts
- [Tesla Optimus](https://x.com/Tesla_Optimus/status/1925047336256078302): Learn from human demonstrations

## Course Focus: Universal Robotics Principles

In this course, we'll be working with the SO100 arm from LeRobot, but the principles, systems, and pipelines we'll explore apply universally across robotics applications:

- Humanoid robots
- Self-driving vehicles
- Autonomous defense systems & many more

The fundamental challenges of perception, planning, control, and learning transfer across all these domains. By mastering these core concepts with our robotic arm, you'll develop skills applicable to the entire field of autonomous systems.

Follow the [get started guide](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md) on so100/101 from Lerobot to assemble, calibrate, and test the robot.
Collect your [first episodes](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#j-train-a-policy) and train your first model.

## The Evolution of Robotics: From Factory Arms to Intelligent Systems

Traditional robotics has been dominated by industrial and factory robots for decades. These machines excel at repetitive, precisely defined tasks in controlled environments. A typical factory robot might:

- Weld the same joint thousands of times per day
- Pick and place identical objects from conveyor belts
- Perform precise cuts or assemblies with sub-millimeter accuracy

However, these robots are fundamentally "dumb" - they follow explicit programming without understanding their environment. They require:

- Perfectly structured environments
- Hard-coded motion sequences
- Safety cages to separate them from humans
- Constant reprogramming for new tasks

The introduction of collaborative robots (cobots) improved flexibility and safety, allowing humans and robots to work together, but these systems still lack true intelligence and adaptability.

## The AI Revolution in Robotics

AI-driven robotics represents a paradigm shift in how robots interact with the world. Instead of explicitly programming every movement, modern approaches allow robots to:

- Perceive and understand their environment through computer vision
- Adapt to changing conditions and novel objects
- Learn from demonstrations and experiences
- Make decisions with uncertainty and incomplete information

Key breakthroughs driving this revolution include:

1. **Deep Learning for Computer Vision**: Transforming camera input into semantic understanding (Krizhevsky et al., 2012 - AlexNet)
2. **Reinforcement Learning**: Enabling robots to learn optimal behaviors through trial and error (OpenAI's Dactyl, 2018)
3. **Imitation Learning**: Learning from human demonstrations (BC-Z, 2023)
4. **Vision-Language Models**: Connecting visual perception with language understanding (RT-2, 2023)
5. **Foundation Models for Robotics**: Leveraging large pre-trained models for transfer learning in robotics (RT-X, 2023)

## Data Augmentation: Then and Now

Data has always been the lifeblood of AI systems, but collecting robot data is expensive and time-consuming. Data augmentation techniques help maximize the utility of existing datasets.

### Traditional Computer Vision Augmentation

Classic data augmentation in computer vision relied on simple transformations:
- Random rotations and flips
- Zooming and cropping
- Color jittering
- Noise addition

These techniques create variations of existing training samples, improving model robustness and generalization.

### Modern Generative Approaches

With generative AI, we can now create entirely new training examples:
- Inpainting missing parts of scenes
- Creating novel viewpoints with NeRF (Neural Radiance Fields)
- Generating synthetic training data with diffusion models
- Domain randomization for sim-to-real transfer

[![Data Augmentation Evolution](assets/data_aug.mov)](assets/data_aug.mov)

## The Robotics Paradox

One of the fascinating paradoxes in AI is that tasks humans find difficult (like playing chess or generating text) have proven easier for AI to master than tasks toddlers can do effortlessly (like manipulating objects or walking).

Robotic manipulation remains challenging despite advances in language models because:

1. **The Physical World is Complex**: Dealing with physics, friction, and deformable objects
2. **High-Dimensional Continuous Control**: Precisely controlling many degrees of freedom simultaneously
3. **Sparse Rewards**: Success often provides binary feedback (success/failure) rather than a gradient to follow
4. **Limited Data**: Language models train on billions of examples; robot datasets are much smaller

### The Data Advantage

Companies with access to large fleets of robots or vehicles have a significant competitive advantage:

- **Tesla**: Collects data from millions of vehicles, enabling rapid improvement in autonomous driving
- **Comma.ai**: Leverages a growing fleet of retrofit vehicles for autonomous driving development
- **Google**: Investing heavily in robot data collection across various platforms and environments

## From Human Demonstration to Robot Execution

> The current trend in human-demos-to-robot-policy papers reminds me of a time when many sim-to-real papers were focused on a particular theme:
> - You train a policy on the sim domain.
> - Then, during inference, you perform a domain transfer from real-to-sim to ensure consistency with the training distribution.
>
> Now, papers like R+X and others are doing something very similar:
> - They train on data from human hands.
> - Then, during inference, they output a hand pose and map it (via IK, etc.) to the robot's pose.
> - Simple yet effective.
>
> Because of the limitations of this mapping approach, we will see more methods involving the co-training of human and robot data. Approaches like PH²D/HAT or DreamGen (NVIDIA's GROOT) treat humans as just another embodiment to learn from.
>
> As VLMs mature, we are seeing the emergence of sim-to-real working beautifully (Proc4Gem) in co-training setups (with sim-to-real batch ratios moving from 1:9 to almost 1:99 - recipe from NVIDIA). This is thanks to VLM's strong semantic grounding; trained on massive image datasets, modern VLMs have an excellent understanding of semantics and spatial reasoning.
>
> As we get into video models grounded in physics, might we see a similar emergence in the human-demo-to-robot-policy space? I think we are two to three years away from a point where a robot can learn directly from YouTube videos.

## Vision Language Models and Sim-to-Real Transfer

> Using current VLMs - think Gemini or Gemma 3 and friends - kinda brings us very close to tackling the sim-to-real issue.
>
> Semantic Grounding: the current VLMs are trained on massive amount of images, they have really good understanding of semantics and spatial reasoning.
>
> Regularization Effect? The VLM's strong priors might also act as a kind of regularizer during sim training. So, even if there is some distribution gap between sim and real world, unlike a model trained from scratch on sim data, a VLM-based policy might not overfit for sim domain.
>
> Bridging the Physics Gap? this will likely only get better as we go from VLMs to foundational world models or even video models (for e.g., Veo 2 & friends). These models build more implicit knowledge of physics, object permanence, and causality. A policy built on such a model might be even more robust to sim-to-real gaps, especially in dynamics, because its internal "world model" is more aligned with reality, potentially requiring less perfect simulation fidelity.
>
> For now, the quality and diversity of the simulation still matter immensely, especially for contact-rich tasks where physics fidelity is key. You still need good simulation. The Proc4Gem paper highlights this too – they addressed all three components: using MuJoCo for high-fidelity physics, Unity for photorealistic rendering, and procedural generation for diverse scenes.
>
> Leveraging large pre-trained models drastically changes the sim-to-real equation, making simulation a much more powerful tool for robotics than ever before. It reduces the reliance purely on real-world data and makes the transfer process more robust, largely thanks to the models' pre-existing world knowledge.
> It's a very exciting time for simulation in robotics, or synthetic data in general!
