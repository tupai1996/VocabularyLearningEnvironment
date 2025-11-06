from django import forms
from .models import Member, QuizList, StudySession, VocabularyList
from django.db.models import Q

class MemberForm(forms.ModelForm):
  password = forms.CharField(widget=forms.PasswordInput)  

  class Meta:
    model = Member
    fields = ["username", "password"]
    
class StudySessionForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None) 
        super().__init__(*args, **kwargs)
        if user is not None:
            qs = VocabularyList.objects.filter(Q(user=user) | Q(is_public=True)).distinct()
        else:
            qs = VocabularyList.objects.filter(is_public=True)

        self.fields["vocabulary_list"].queryset = qs
        self.fields["name"].widget.attrs["placeholder"] = "Session name"
        self.fields["goal_value"].widget.attrs["placeholder"] = "Goal value"
        self.fields["start_date"].widget.attrs["placeholder"] = "Start date"
        self.fields["end_date"].widget.attrs["placeholder"] = "End date"

    class Meta:
        model = StudySession
        fields = ["name", "vocabulary_list", "quiz_list", "goal_type", "goal_value", "start_date", "end_date"]
        widgets = {
            "start_date": forms.DateInput(attrs={"type": "date"}),
            "end_date": forms.DateInput(attrs={"type": "date"}),
        }

