# Generated by Django 3.2.9 on 2021-11-21 05:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_disease_pathogen'),
    ]

    operations = [
        migrations.AlterField(
            model_name='solution',
            name='solution',
            field=models.CharField(max_length=255),
        ),
    ]