# S12 Readme
==================================
## PART 1 - Open CV Yolo model run
==================================

I took an image of myself along with a book and ran the standard opencv yolo model.
It was able to identify me (person), and the book, as well as the image of the bird on the cover on the book.

![image](https://user-images.githubusercontent.com/498461/229636492-d12c632c-f066-4ab4-a394-081e4ba4f806.png)

==========================================================
## PART 2 - Training Yolo V3 with custom images
==========================================================

Took around 25 images of the Korean pop group BLACKPINK and annotated all the 4 members of the group.
However, during testing, found out model was having difficulty with identifying similar faces (especially two members who are very similar).
The final output was very disappointing even after running it for 300 epochs.

Hence, moved onto a cartoon screenshots of Kung Fu Panda movie.
The outputs are better and the training shots are like this.
![image](https://user-images.githubusercontent.com/498461/229637648-9f03a2d7-b094-4415-894c-9a4d3613312d.png)

The test set images are like this...
![image](https://user-images.githubusercontent.com/498461/229640076-43bc44ff-00ec-45c7-a6db-bed271c32fc2.png)

The validation set images are like
![image](https://user-images.githubusercontent.com/498461/229639625-dfa5e642-d947-4e42-9921-cd08f3a72b4b.png)

==========================================================
## PART 3 - Running of a video against the trained model
==========================================================

Kung fu panda movie trailer was downloaded, all the images were extracted using ffmpeg.
Then the detect.py was run on all the images (~3K) and then they were all merged together as a single video.
Even the audio is added back.

The video can be found at https://youtu.be/hZ1CyQYnQig
