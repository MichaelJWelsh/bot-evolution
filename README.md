# Bot Evolution
Bot evolution is an interesting display of evolution through neural networks and a genetic algorithm. Bot's have a field of vision represented by their antenna's. They are told if they can "see" food in their field of vision, and are then asked to either move forward, turn counterclockwise, turn clockwise, or do nothing. If a bot does not recieve food after a certain period of time, it will die off. When a bot gets food, it reproduces asexually with a chance of mutation. If a bot goes too far out of the map, it dies and a completely random bot is spawned in the middle of the map. Each bot has its own neural network. You can see species emerge based on their colors.


![](https://github.com/MichaelJWelsh/bot-evolution/blob/master/example.gif)


## Usage
Simply go into the source folder and type this in terminal:
```python
python3.5 main.py
```


## Dependencies
 - numpy
 - pygame
