---
excerpt: algorithms visualization with JavaFX
author_profile: true
title:  "Algorithm Visualization Web App with JavaFX, jpro, Docker and Heroku"
categories:
  - coding
tags:
  - web api
header:
  overlay_image: /assets/images/algo.png
  teaser: /assets/images/algo.png
  overlay_filter: 0.5
---
I was looking for a project idea to help me practice Java programming and software development. The project idea should be simple enough so that I can concentrate on the software design and development aspect of the program. Besides, it should serve my own needs and interests. Then I came across the [Algorithm Visualization Project](https://clementmihailescu.github.io/Pathfinding-Visualizer/) (JavaScript) and was fascinated by the project idea, which perfectly combines my need to revisit algorithms, to develop a software and practice making frontends. Therefore, I decided to make my own algorithm visualizer in Java.

A natural question arises, how to develop the front end part? There are many ways to do that, for example using JavaScript + HTML + CSS. Since the goal of the project is to practice Java, I decided to go with the JavaFX (with FXML and CSS) for the user interface. A small disadvantage of JavaFX is that it is mostly used for desktop applications and not really suitable to serve remotely. This requires the users to download and run the application locally. I prefered a cloud solution, so I employed the [JPro](https://www.jpro.one/) to make the application available via browsers. Next, I follow the example [JPro on Heroku](https://github.com/FlorianKirmaier/JPro-Heroku) to deploy the application to Heroku.

Visit a demo of the application here: [Algorithm Visualizer](https://algorithm-visualizer-javafx.herokuapp.com/)

![png](/assets/images/algo.png)

I have implemented 3 types of basic algorithms: sorting, path finding and array searches. The users can generate random examples and hit Start to see the visualization:
![gif](/assets/images/algo/graph_cropped.gif)
![gif](/assets/images/algo/search_cropped.gif)
![gif](/assets/images/algo/sort_cropped.gif)


For the project, I decided to follow the Model-View-Controller pattern. 
