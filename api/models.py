from django.db import models


class Disease(models.Model):
    """ DiseaseModel - model to store disease details """

    name = models.CharField(null=False, max_length=255, unique=True)
    pathogen = models.CharField(max_length=255)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        self.name = self.name.replace(' ', '_').lower()
        super().save(*args, **kwargs)


class Symptom(models.Model):
    """ SymptomModel - model to store symptoms"""

    disease = models.ForeignKey(Disease, null=False, on_delete=models.CASCADE)
    symptom = models.CharField(null=False, max_length=255)

    def __str__(self):
        return '{} symptom {}'.format(self.disease, self.id)


class Solution(models.Model):
    """ SolutionModel - model to store treatments for a disease """

    disease = models.ForeignKey(Disease, null=False, on_delete=models.CASCADE)
    solution = models.CharField(null=False, max_length=255)

    def __str__(self):
        return '{} solution {}'.format(self.disease, self.id)
