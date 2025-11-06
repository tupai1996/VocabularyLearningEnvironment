from django.contrib import admin
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from django.contrib.admin.models import LogEntry

from .models import (
    Vocabulary, Member, VocabularyList, UserAnswer,
    UserMemory, StudySession, DailyReviewCounter, DailyMinuteCounter, QuizList, QuizHistory
)

admin.site.register(Vocabulary)
admin.site.register(Member)
admin.site.register(VocabularyList)
admin.site.register(UserAnswer)
admin.site.register(UserMemory)
admin.site.register(StudySession)
admin.site.register(DailyReviewCounter)
admin.site.register(DailyMinuteCounter)
admin.site.register(QuizList)
admin.site.register(QuizHistory)

class SuperuserOnlyAdmin(admin.ModelAdmin):
    def _ok(self, request):
        return request.user.is_active and request.user.is_superuser

    def has_module_permission(self, request):
        return self._ok(request)

    def has_view_permission(self, request, obj=None):
        return self._ok(request)

    def has_add_permission(self, request):
        return self._ok(request)

    def has_change_permission(self, request, obj=None):
        return self._ok(request)

    def has_delete_permission(self, request, obj=None):
        return self._ok(request)


for model in (Group, Permission, ContentType, Session, LogEntry):
    try:
        admin.site.unregister(model)
    except admin.sites.NotRegistered:
        pass
    admin.site.register(model, SuperuserOnlyAdmin)
