# Generated by Django 3.2.9 on 2021-11-21 04:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='DiseaseModel',
            new_name='Disease',
        ),
        migrations.RenameModel(
            old_name='SolutionModel',
            new_name='Solution',
        ),
        migrations.RenameModel(
            old_name='SymptomModel',
            new_name='Symptom',
        ),
    ]
