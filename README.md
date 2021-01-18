# A face that turns into the Pikachu

### Face detector with face_landmark

![fianl](https://user-images.githubusercontent.com/71427403/104696147-2afb1e80-5751-11eb-8ae1-85d21b9d0567.png)


[Video](https://youtu.be/y6sAJbscK-w)
---
## Method

1. Download [shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) and put in model folder

2. Set **save Path** in ```result_path```

3. Run final_result.py

---
## Workflow

1. Find face with ```rectangle``` in video   
2. Add face_landmarks by ```dlib.shape_predictor```   
3. Compute center and boundaris of face   
4. ```overlay_transparent``` to add Pikachu   
