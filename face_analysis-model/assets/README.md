# assets/

Static files shipped with the application (icons, sample cast photos, demo clips).

Nothing here is required to run: the GUI styles itself from
[gui/main_window.py](../gui/main_window.py) and model weights are downloaded into
`models/weights/` on first use.

Suggested layout if you add material:

```
assets/
  icons/            application and toolbar icons
  cast_samples/     example reference photos, one folder per actor
  clips/            short videos used for smoke tests and demos
```

A folder of reference photos here can be enrolled directly:

```bash
python cli.py register --name "John" --images assets/cast_samples/john
```
