class ByteTrackArgument:
    track_thresh = 0.5 # High_threshold
    track_buffer = 50 # Number of frame lost tracklets are kept
    match_thresh = 0.8 # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0 # Minimum bounding box aspect ratio
    min_box_area = 1.0 # Minimum bounding box area
    mot20 = False # If used, bounding boxes are not clipped.

MIN_THRESHOLD = 0.001