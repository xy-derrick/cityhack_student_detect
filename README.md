# City Hack 2023

## Team

team name:rp++ 

members: 

XIA Shujun (Leader) 

XU Yuan 

NG Cheuk Yiu 

CHEN Yihuan  

Ng Pak Lam

## Probelm description

In the foreseeable future, online classes will continue to be used as a complementary means of teaching and learning. One of the disadvantages of online classes is that the teacher cannot be in front of the students as in an offline classroom and thus follow the students' expressions to see how they are doing. When a student is confused and not keeping up, the teacher cannot detect it in real-time. Also, students are easily distracted and it’s hard to be monitored by the course instructors online. Such a situation calls for a mechanism to detect the expression of students to return real-time feedback to teachers and thus improve class interaction.  

## Solution

Our idea is to develop a real-time facial expression detection tool which analyzes facial attributes to calculate the confusion or attention level of students and give feedback to teachers. 

## Workflow

- Input video sequence of student
- Extract features like head pose, eye direction, lip movement etc.
- Calculate the confusion/attention level using the deviations
- Return the feedback

## Data Processing and Algorithms

sleepy count: 

Blinking count: eye aspect ratio of 0.2 for 3 consecutive frames 

Yawning count: mouth aspect ratio of 0.5 for 3 consecutive frames 

Sleepy nod count: pitch(x) rotation angle of 0.3 for 3 consecutive frames 

Confusion count: 

For the first 259 consecutive frames: 

The algorithm takes the distance between the eyebrows for each frames and calculates the average distance for 259 frames. 

Starting from the 260 frame: 

The algorithm compares whether the distance of eyebrows of each frame is shorter than the average. If yes, the confusion count increases by one. 

## Impact

By processing the data of each student to obtain an overall level, we can know the learning effectiveness of the whole class, which provides a reference for teachers to improve teaching methods.

This system helps teachers comprehensively understand students’ attentiveness and improve the teaching quality by promptly tracing  students’ situations.
