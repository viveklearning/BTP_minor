        # Process every nth frame based on frame rate
        if frame_count % frame_rate == 0:
            # Process Visual Stream
            visual_emotion = get_visual_emotion(frame)