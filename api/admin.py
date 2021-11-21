from django.contrib import admin
from api.models import *


@admin.register(Symptom)
class SymptomAdmin(admin.ModelAdmin):
    """ Symptom admin """

    list_display = ['id', 'disease', 'symptom']
    search_fields = ['id', 'disease__name', 'symptom']


@admin.register(Solution)
class SolutionAdmin(admin.ModelAdmin):

    list_display = ['id']


class SymptomInline(admin.StackedInline):
    """ Inline for symptoms """

    model = Symptom
    extra = 0


class Solution(admin.StackedInline):
    """ Inline for solutions """

    model = Solution
    extra = 0


@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    """ Disease admin """

    list_display = ['id', 'name']
    search_fields = ['id', 'name']
    inlines = [SymptomInline, Solution]