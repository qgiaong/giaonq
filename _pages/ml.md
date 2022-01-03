---
title: "Machine Learning Projects"
layout: pages
permalink: /machine-learning/

author_profile: true
---
Some selected machine learning projects
{% include group-by-array.html collection=site.posts field='machine learning' %}

<ul>
  {% for tag in group_names %}
    {% assign posts = group_items[forloop.index0] %}

    <li>
      <h2>{{ tag }}</h2>
      <ul>
        {% for post in posts %}
        <li>
          <a href='{{ site.baseurl }}{{ post.url }}'>{{ post.title }}</a>
        </li>
        {% endfor %}
      </ul>
    </li>
  {% endfor %}
</ul>
