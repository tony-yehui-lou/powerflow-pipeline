## Runtime
## UI
1. Graphs
   1. Bar Height Curve 
   2. Bar Velocity Curve 
   3. Bar Acceleration Curve 
2. Video
   1. Lift Video
   2. Circle-stick labelling of joints + bones
   3. Barbell labelling
   4. COM labelling (both barbell and person)
   5. Stage Splitting
   6. Labelled issues on video
3. Feedback
   1. Text feedback on issues
4. Interface type
   1. Website
   2. iPhone App

### Graphs
#### Bar Path Curve
Input: 
1. A smoothed function, 
2. labels for x,y axis, 
3. marks on x-axis determing stage
Output: 
1. A graph depicting the function, with xy axis
2. Translucent boxes on sections depicting which stage of the lift it is
#### Bar Velocity Curve
Input: 
1. A smoothed function, 
2. labels for x,y axis, 
3. marks on x-axis determing stage
Output: 
1. A graph depicting the function, with xy axis
2. Translucent boxes on sections depicting which stage of the lift it is
3. Maximal point labelled, and data will appear upon clicking
#### Bar Acceleration Curve
Input: 
1. A smoothed function, 
2. labels for x,y axis, 
3. marks on x-axis determing stage
Output: 
1. A graph depicting the function, with xy axis
2. Translucent boxes on sections depicting which stage of the lift it is
3. Maximal point labelled, and data will appear upon clicking

### Video
#### Lift Video
Input: 
1. video.mp4
Output:
2. A rectangular box depicting video.mp4
#### Circle-stick labelling of joints + bones
Input: 
1. pixel position of joints
2. radius of circle
Output:
1. Translucent circles drawn on the respective positions, which is an overlay over the previous videomp4 box
2. Lines connecting each circle from centre to centre
#### Barbell labelling
Input:
1. start and end pixel of barbell
Output:
1. Line drawn from start to end of barbell, overlayed on same lift video box.
#### COM labelling (both barbell and person)
Input:
1. Two pixels
2. Radius
Output:
1. Two circles centred at the pixels with radius, overlayed on same lift video box.
#### Stage Splitting
Input:
1. Time stamps
Output:
1. When video is played, change colour of all overlayed labelling in a fixed order that is written into code.
#### Labelled issues on video
Input:
1. Tuples of issue indices with pixels with radius.
2. Issue array
Output:
1. At each corresponding pixel, draw a circle with corresponding radius.
2. When the circle is clicked on, it scrolls to the corresponding text feedback.
### Feedback
#### Text feedback on issues
Input:
1. Issue indices
2. Issue array
Output:
1. Rectangular boxes drawn below the video box and graph, with a title writing the issue, and then a body text writing the specific details. These are all found in the issue array.
2. Append a youtube video at the end that can be watched directly. The link can be found in issue array.
