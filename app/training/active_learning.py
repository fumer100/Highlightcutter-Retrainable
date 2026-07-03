class ActiveLearningLoop:

    def __init__(self, trainer):

        self.trainer = trainer

    def run_weekly(self):

        print("🧠 Starte wöchentliches Retraining")

        self.trainer.train(
            epochs=30,
            img_size=640
        )