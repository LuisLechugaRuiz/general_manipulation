**Lessons Learned:**

- **Visualization is Key**: Examining attention weights over images has been invaluable. Not only does it make debugging more straightforward, but it also paves the way for innovative ideas.

- **Real-time Predictions with Transformers? A Challenge**: Real-time predictions using transformers is quite the task. Take Nvidia's RVT for instance. They achieved notable results by training on discrete frames (a.k.a keyframes), which allowed their models to grasp transitions. However, this approach has its limitations. It's not scalable in dynamic environments where objects are in motion or when obstacles come into play. And when you try to process all frames? New set of problems. The model struggles with transitions between highly similar frames. And if it needs to revert to a previous keypoint? There's uncertainty about its position in the task sequence. It's becoming clearer that a memory system might be a game changer for real-time processing.

- **Dual Decoders for the Win**: Employing two decoders seems to be a productive move. My lightbulb moment was when I observed the attention overlay of two separate models. It just clicked that a shared encoder was the route forward. Results so far? Promising, though there's room for improvement. And yes, the training dynamics? Quite unique.

- **Leveraging Nvidia's Checkpoint**: I'm currently working off Nvidia's checkpoint, which, as mentioned, banks on keyframes. Tweaked the dataset a bit to filter out stationary manipulator samples (don't want the joint positions predicting the same moves repeatedly) and to assess all frames. Full training is on the horizon, but it's a time-intensive endeavor, especially with my RTX4080 setup.

- **The Magic of Temporal Ensemble**: This technique has proven instrumental in predicting subsequent joint positions. And while I haven't incorporated the CVAE (for style) yet, it's on my radar, especially for human demo training.
