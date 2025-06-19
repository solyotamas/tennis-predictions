#  Tennis Match Predictions

A tennis match prediction project focusing on supervised machine learning algorithms — **Decision Trees** and **Random Forests**.

I was inspired by a video explaining Decision Trees and Random Forests, and decided to implement my own versions in pure Python to better understand how these algorithms work.

Since I'm really into sports, I also wanted to explore feature engineering techniques, especially using **Elo-based features**, to see how they can improve sports predictions and how closely they can represent the real strength of the players.

For the tennis data, I used the famous and very detailed **Jeff Sackmann tennis dataset** from GitHub.

---

##  References / Inspirations

**Really detailed, holy grail of tennis datasets:**
- Jeff Sackmann Tennis Data (ATP), licensed under CC BY-NC-SA 4.0: [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)

**Found a really detailed and cool study about Weighted Elo and rating predictions:**
- Angelini, G., Candila, V., & De Angelis, L. (2022). *Weighted Elo rating for tennis match predictions*. European Journal of Operational Research, 297(1), 120–132. [DOI: 10.1016/j.ejor.2021.04.011](https://doi.org/10.1016/j.ejor.2021.04.011)

**The video that inspired me:**
- [YouTube Video](https://www.youtube.com/watch?v=LkJpNLIaeVk)

---

##  Project Goal, Hardships, and Future Improvements

### Project Goal
The main goal of this project was to apply Decision Trees and Random Forests to a real-world dataset while having some fun implementing player elo in a sports game.

### Hardships
- While it's nice to build own implementations, for datasets in the ten thousands it is painfully slow without using numpy or underlying C code, so I ultimately used scikit-learn's version.
- Other challenging thing to implement was:
  - Semi-inactive players
  - Retired players
  - Players who mostly play on specific surfaces

### Future Improvements
- Add 2025 matches to the dataset (currently includes matches up to 2024).
- Improve the Elo modeling logic to better handle edge cases.
- Explore modeling another sport that's more team-oriented.

---
