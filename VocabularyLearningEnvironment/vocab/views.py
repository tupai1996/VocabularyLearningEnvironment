import random
import time
import unicodedata
import re
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.db import transaction, OperationalError
from django.db.models import F
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from components.teacher.items import WordItem
from components.learners.exp_memory import ExpMemoryLearner
from components.teacher.planners import RandomPlanner

from .forms import MemberForm, StudySessionForm
from .models import (Member,QuizList,UserAnswer,UserMemory,Vocabulary,VocabularyList,StudySession,DailyReviewCounter,ActiveStudySession,DailyMinuteCounter,QuizHistory)

planner = RandomPlanner()

def _reset_quiz_flags(user, quiz_list_id: int):
    UserMemory.objects.filter(user=user,vocabulary__quiz_list_id=quiz_list_id,is_asked_in_quiz=True).update(is_asked_in_quiz=False)

@login_required
def session_info(request, session_id):
    s = get_object_or_404(StudySession, id=session_id, user=request.user)
    return JsonResponse({"goal_type": s.goal_type})


def _normalize(s: str):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def main_page(request):
    return render(request, "vocab/main_page.html")


@login_required
def user_page(request):
    member = request.user
    username = member.username

    user_decks = VocabularyList.objects.filter(user=member)
    public_decks = VocabularyList.objects.filter(is_public=True).exclude(user=member)

    if request.method == "POST" and request.POST.get("form_type") == "create_session":
        session_form = StudySessionForm(request.POST, user=member)
        if session_form.is_valid():
            session = session_form.save(commit=False)
            session.user = member

            if session.goal_type == "quiz":
                available_count = (Vocabulary.objects.filter(usermemory__user=member).distinct().count())
                if available_count < session.goal_value:
                    session_form.add_error(
                        "goal_value",
                        f"Not enough words in memory. You have {available_count} words learned, "
                        f"but requested {session.goal_value} questions."
                    )
                    messages.error(request, "Could not create quiz session.")
                    return render(request, "vocab/user_page.html", {
                        "username": username,
                        "user_decks": user_decks,
                        "public_decks": public_decks,
                        "session_form": session_form,
                        "sessions": StudySession.objects.filter(user=member).order_by("-created_at"),
                        "quiz_sessions": StudySession.objects.filter(user=member, goal_type="quiz").order_by("-created_at"),
                    })
                with transaction.atomic():
                    quiz_list = create_quiz_list(user=member, question_count=session.goal_value)
                    quiz_list.name = session.name or ""            
                    quiz_list.save(update_fields=["name"])   
                    session.quiz_list = quiz_list
                    session.save()

                messages.success(request, f"Quiz session created with {session.goal_value} questions.")
            else:
                session.save()
                messages.success(request, "Study session created.")

            return redirect("user_page")
    else:
        session_form = StudySessionForm(user=member)

    sessions = (StudySession.objects
                .filter(user=member)
                .exclude(goal_type="quiz")
                .select_related("vocabulary_list")
                .order_by("-created_at"))

    quiz_sessions = (StudySession.objects
                     .filter(user=member, goal_type="quiz")
                     .select_related("vocabulary_list")
                     .order_by("-created_at"))

    return render(
        request,
        "vocab/user_page.html",
        {
            "username": username,
            "user_decks": user_decks,
            "public_decks": public_decks,
            "session_form": session_form,
            "sessions": sessions,           
            "quiz_sessions": quiz_sessions,  
        },
    )
def get_public_decks(request):
    member = request.user
    public_decks = VocabularyList.objects.filter(is_public=True).exclude(user=member)

    decks_data = [
        {
            "id": deck.id,
            "list_name": deck.list_name,
            "description": deck.description,
            "creator": deck.user.username,
        }
        for deck in public_decks
    ]
    return JsonResponse({"decks": decks_data})


@ensure_csrf_cookie
def home(request):
    return render(request, "vocab/home.html")

def choose_random_word(user, session):
    if session.quiz_list:
        with transaction.atomic():
            deck = QuizList.objects.select_for_update().get(pk=session.quiz_list.id)

            vocab_qs = Vocabulary.objects.filter(
                quiz_list=deck,
                usermemory__user=user,
                usermemory__is_asked_in_quiz=False
            )

            if not vocab_qs.exists():
                return {
                    "status": "done",
                    "message": "Quiz complete.",
                    "score": deck.score,
                    "total": deck.question_count
                }

            chosen_vocab = vocab_qs.order_by("?").first()

            UserMemory.objects.filter(user=user, vocabulary=chosen_vocab).update(
                is_asked_in_quiz=True
            )

        return {
            "status": "ok",
            "word": chosen_vocab.source_word,
            "translation": chosen_vocab.target_word,
            "question_id": chosen_vocab.id
        }
    deck = session.vocabulary_list
    vocab_qs = Vocabulary.objects.filter(vocabulary_list=deck)

    if not vocab_qs.exists():
        return {"status": "error", "message": "This deck is empty."}

    item_list = []
    vocab_dict = {}
    for vocab in vocab_qs:
        word_item = WordItem(source=vocab.source_word, target=vocab.target_word)
        item_list.append(word_item)
        vocab_dict[word_item] = vocab

    chosen_item = planner.choose_item(item_list, context=None, time=0)
    chosen_vocab = vocab_dict[chosen_item]
    question = chosen_item.get_question()
    translation = chosen_item.get_answer()

    now_seconds = int(timezone.now().timestamp())
    learner = ExpMemoryLearner.load_memory_from_db(user, alpha=0.1, beta=0.5)
    learner.learn(chosen_item, chosen_vocab.id, now_seconds)
    learner.save_memory_to_db_with_retry(user)

    return {
        "status": "ok",
        "word": question,
        "translation": translation,
        "question_id": chosen_vocab.id
    }


@login_required
def random_word_view(request):
    member = request.user
    session_id = request.GET.get("session_id")
    if not session_id:
        return JsonResponse({"status": "error", "message": "session_id is required."})

    session = get_object_or_404(
        StudySession.objects.select_related("vocabulary_list"),
        id=session_id,
        user=member,
    )
     
    data=choose_random_word(member, session)
    return JsonResponse(data) 


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            messages.success(request, "Welcome back!")
            return redirect("user_page")
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, "vocab/login.html", {})
    else:
        return render(request, "vocab/login.html", {})


def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect("main_page")


def join(request):
    if request.method == "POST":
        form = MemberForm(request.POST)
        if form.is_valid():
            member = form.save(commit=False)
            member.set_password(form.cleaned_data["password"])
            member.save()

            messages.success(request, "Account created.")
            return redirect("login")
        else:
            messages.error(request, "Username already taken or invalid form.")
            return render(request, "vocab/join.html", {"form": form})
    else:
        form = MemberForm()
    return render(request, "vocab/join.html", {"form": form})


@login_required
def create_list(request, count):
    if request.method == "POST":
        member = request.user
        if member:
            list_name = request.POST.get("list_name")
            description = request.POST.get("description")
            is_public = request.POST.get("is_public")

            new_deck = VocabularyList.objects.create(
                list_name=list_name,
                description=description,
                user=member,
                is_public=bool(is_public),
            )

            word_items = planner.load_chosen_words(count, user=member)
            for item in word_items:
                Vocabulary.objects.create(
                    source_word=item.source,
                    target_word=item.target,
                    source_language="en",
                    target_language="de",
                    vocabulary_list=new_deck,
                )

    return redirect("user_page")


@login_required
def delete_list(request, list_id):
    if request.method == "POST":
        member = request.user
        deck = get_object_or_404(VocabularyList, id=list_id, user=member)
        name = deck.list_name
        deck.delete()

        messages.success(request, f'"{name}" deleted.')

    return redirect("user_page")


def _is_correct(given: str, expected: str) -> bool:
    g = _normalize(given)
    e = _normalize(expected)

    if not g or g == "":
        return False

    if g == e:
        return True

    rm = lambda x: re.sub(r"[^\w\s]", "", x)
    return rm(g) == rm(e)



@require_POST
@login_required
@transaction.atomic
def submit_answer(request):
    user = request.user
    question_id = request.POST.get("question_id")
    given_answer = request.POST.get("given_answer", "")

    if not (user and question_id):
        return JsonResponse({"status": "error", "message": "Invalid request"})

    question = get_object_or_404(Vocabulary, id=question_id)

    expected = question.target_word
    correct = _is_correct(given_answer, expected)

    quiz = question.quiz_list
    if not quiz:
        user_answer = UserAnswer.objects.create(
            user=user,
            question=question,
            quiz_list=None,
            given_answer=given_answer,
            is_correct=correct,
        )
        return JsonResponse({"status": "ok", "saved_id": user_answer.id, "is_correct": correct})

    deck = QuizList.objects.select_for_update().get(pk=quiz.pk)

    user_answer = UserAnswer.objects.create(
        user=user,
        question=question,
        quiz_list=deck,
        given_answer=given_answer,
        is_correct=correct,
    )

    update_fields = {"asked_count": F("asked_count") + 1}
    if correct:
        update_fields["score"] = F("score") + 1
    QuizList.objects.filter(pk=deck.pk).update(**update_fields)
    deck.refresh_from_db(fields=["asked_count", "score", "question_count"])

    if deck.asked_count >= deck.question_count:
        last = QuizHistory.objects.filter(user=user, name=request.POST.get("session_name")).order_by('-attempt').first()
        attempt_number = (last.attempt if last else 0) + 1
        if not QuizHistory.objects.filter(user=user, name=request.POST.get("session_name"), attempt=attempt_number).exists():
            QuizHistory.objects.create(
                user=user,
                score=deck.score,
                question_count=deck.question_count,
                attempt=attempt_number,
                name=request.POST.get("session_name")
            )
        _reset_quiz_flags(user, deck.id)

    return JsonResponse({
        "status": "ok",
        "saved_id": user_answer.id,
        "is_correct": correct,
        "score": deck.score,
        "asked_count": deck.asked_count,
        "total": deck.question_count,
        "done": deck.asked_count >= deck.question_count
    })



@login_required
#Conveys input form to the session model
def study_sessions(request):
    member = request.user

    if request.method == "POST":
        form = StudySessionForm(request.POST, user=member)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = member

            if session.goal_type == "quiz":
                try:
                    with transaction.atomic():
                        quiz_list = create_quiz_list(user=member, question_count=session.goal_value)
                        session.quiz_list = quiz_list
                        session.save()
                except ValueError as e:
                    form.add_error("goal_value", str(e))
                    messages.error(request, "Could not create quiz session.")
                    sessions = StudySession.objects.filter(user=member).order_by("-created_at")
                    return render(request, "vocab/study_sessions.html", {"form": form, "sessions": sessions})
            else:
                session.save()

            messages.success(request, "Session created.")
            return redirect("study_sessions")
    else:
        form = StudySessionForm(user=member)

    sessions = StudySession.objects.filter(user=member).order_by("-created_at")
    return render(request, "vocab/study_sessions.html", {"form": form, "sessions": sessions})

@login_required
def start_session(request, session_id):
    member = request.user
    session = get_object_or_404(StudySession, id=session_id, user=member)

    if not session.is_running_today():
        return JsonResponse(
            {
                "status": "error",
                "message": "This study session has ended or has not started yet.",
            },
            status=400,
        )

    if session.goal_type != "quiz":
        has_words = Vocabulary.objects.filter(
            vocabulary_list=session.vocabulary_list
        ).exists()
        if not has_words:
            return JsonResponse({"status": "error", "message": "This deck is empty."})

    if session.goal_type == "reviews_per_day":
        today = timezone.localdate()
        counter, _ = DailyReviewCounter.objects.get_or_create(
            user=member,
            study_session=session,
            date=today,
            defaults={"count": 0},
        )
        DailyReviewCounter.objects.filter(pk=counter.pk).update(count=F("count") + 1)

    return JsonResponse({"status": "started", "session_id": session.id})

@login_required
@require_POST
def reverse_privacy(request, deck_id):
    member = request.user

    deck = get_object_or_404(VocabularyList, id=deck_id, user=member)
    deck.is_public = not deck.is_public
    deck.save()
    return redirect("user_page")


@require_POST
@login_required
@transaction.atomic
#Starts the timer for minutes per day session
def start_study_session(request):
    sid = request.POST.get("study_session_id")
    if not sid:
        return HttpResponseBadRequest("study_session_id is required")

    session = get_object_or_404(StudySession, id=sid, user=request.user)

    if session.goal_type != "minutes_per_day":
        ActiveStudySession.objects.filter(user=request.user).delete()
        return JsonResponse({"status": "skipped", "reason": "not_minutes_goal"})

    ActiveStudySession.objects.filter(user=request.user).delete()
    active = ActiveStudySession.objects.create(
        user=request.user,
        study_session=session,
    )
    return JsonResponse(
        {
            "status": "started",
            "session_id": active.id,
            "started_at": active.started_at.isoformat(),
        }
    )


# Ends the user's active study session, logs any elapsed minutes to today's counter, and removes the active session record.
@require_POST
@login_required
@transaction.atomic
def end_study_session(request):
    active = (
        ActiveStudySession.objects.select_for_update()
        .filter(user=request.user)
        .select_related("study_session")
        .first()
    )
    if not active:
        return JsonResponse({"status": "ended", "minutes_studied": 0})

    if active.study_session.goal_type != "minutes_per_day":
        active.delete()
        return JsonResponse({"status": "ended", "minutes_studied": 0})

    elapsed_sec = int((timezone.now() - active.started_at).total_seconds())
    full_minutes = elapsed_sec // 60

    if full_minutes > 0:
        today = timezone.localdate()
        counter, _ = DailyMinuteCounter.objects.select_for_update().get_or_create(
            user=request.user,
            study_session=active.study_session,
            date=today,
            defaults={"minutes": 0},
        )
        DailyMinuteCounter.objects.filter(pk=counter.pk).update(
            minutes=F("minutes") + full_minutes
        )

    active.delete()

    return JsonResponse({"status": "ended", "minutes_studied": int(full_minutes)})


@login_required
def get_study_time_status(request):
    active = ActiveStudySession.objects.filter(user=request.user).first()
    if not active:
        return JsonResponse({"active": False})

    elapsed_sec = int((timezone.now() - active.started_at).total_seconds())
    return JsonResponse(
        {
            "active": True,
            "started_at": active.started_at.isoformat(),
            "elapsed_seconds": elapsed_sec,
            "elapsed_minutes": elapsed_sec // 60,
        }
    )

@require_POST
@login_required
@transaction.atomic
def update_study_time(request):
    active = (
        ActiveStudySession.objects.select_for_update()
        .filter(user=request.user)
        .select_related("study_session")
        .first()
    )
    if not active:
        return JsonResponse({"active": False})

    if active.study_session.goal_type != "minutes_per_day":
        return JsonResponse({"active": True, "added_minutes": 0, "ignored": True})

    elapsed_sec = int((timezone.now() - active.started_at).total_seconds())
    full_minutes = elapsed_sec // 60
    if full_minutes <= 0:
        return JsonResponse({"active": True, "added_minutes": 0})

    today = timezone.localdate()
    counter, _ = DailyMinuteCounter.objects.select_for_update().get_or_create(
        user=request.user,
        study_session=active.study_session,
        date=today,
        defaults={"minutes": 0},
    )
    DailyMinuteCounter.objects.filter(pk=counter.pk).update(
        minutes=F("minutes") + full_minutes
    )

    active.started_at = active.started_at + timedelta(minutes=full_minutes)
    active.save(update_fields=["started_at"])

    return JsonResponse({"active": True, "added_minutes": full_minutes})


@require_POST
@login_required
@transaction.atomic
def delete_session(request, session_id):
    session = get_object_or_404(StudySession, id=session_id, user=request.user)

    active = (
        ActiveStudySession.objects.select_for_update()
        .filter(user=request.user, study_session=session)
        .first()
    )
    if active:
        if session.goal_type == "minutes_per_day":
            elapsed_sec = int((timezone.now() - active.started_at).total_seconds())
            # if less than a minute the time spent is not saved!!
            full_minutes = elapsed_sec // 60
            if full_minutes > 0:
                today = timezone.localdate()
                counter, _ = (
                    DailyMinuteCounter.objects.select_for_update().get_or_create(
                        user=request.user,
                        study_session=session,
                        date=today,
                        defaults={"minutes": 0},
                    )
                )
                DailyMinuteCounter.objects.filter(pk=counter.pk).update(
                    minutes=F("minutes") + full_minutes
                )
            active.delete()
    if session.goal_type == "quiz" and session.quiz_list_id:
        qid = session.quiz_list_id
        _reset_quiz_flags(request.user, qid)
        Vocabulary.objects.filter(quiz_list_id=qid).update(quiz_list=None)
        
    session.delete()
    return redirect("user_page")


# Returns today's progress for the given StudySession (reviews or minutes) and whether the daily goal has been completed. Its like a getter funct
@login_required
def progress_check(request, session_id):
    session = get_object_or_404(StudySession, id=session_id, user=request.user)
    today = timezone.localdate()

    if session.goal_type == "reviews_per_day":
        counter = DailyReviewCounter.objects.filter(
            user=request.user, study_session=session, date=today
        ).first()
        progress = counter.count if counter else 0
    else:
        counter = DailyMinuteCounter.objects.filter(
            user=request.user, study_session=session, date=today
        ).first()
        progress = counter.minutes if counter else 0

    return JsonResponse(
        {
            "goal_type": session.goal_type,
            "goal_value": session.goal_value,
            "progress": progress,
            "done": progress >= session.goal_value,
            "is_running_today": session.is_running_today(),
        }
    )

def create_quiz_list(user, question_count):
    user_memory_vocabs = Vocabulary.objects.filter(usermemory__user=user, usermemory__is_asked_in_quiz=False).distinct()    
    available_count = user_memory_vocabs.count()

    if available_count < question_count:
        raise ValueError(f"Not enough words in memory. You have {available_count} words learned, but requested {question_count} questions.")
    
    selected_vocabs = random.sample(list(user_memory_vocabs), question_count)
    selected_ids = [v.id for v in selected_vocabs]
    
    with transaction.atomic():
        quiz_list = QuizList.objects.create(user=user,question_count=question_count)
        Vocabulary.objects.filter(id__in=selected_ids).update(quiz_list=quiz_list)
    
    return quiz_list

def save_quiz_to_history(user, quiz_list, session):
    previous_attempts = QuizHistory.objects.filter(user=user, name=session.name).count()
    attempt_number = previous_attempts + 1

    QuizHistory.objects.create(
        user=user,
        score=quiz_list.score,
        question_count=quiz_list.question_count,
        attempt=attempt_number,
        name = session.name
    )

@require_POST
@login_required
@transaction.atomic
def restart_quiz(request, session_id):
    member = request.user
    session = get_object_or_404(StudySession, id=session_id, user=request.user, goal_type="quiz")
    
    old_quiz = session.quiz_list
    if not old_quiz:
        return JsonResponse({"status": "error", "message": "No quiz associated with this session."}, status=400)

    old_quiz.asked_count = 0
    old_quiz.score = 0
    old_quiz.save()

    UserMemory.objects.filter(user=request.user, vocabulary__quiz_list=old_quiz).update(is_asked_in_quiz=False)

    return JsonResponse({"status": "ok","message": "Quiz reset.","quiz_list_id": old_quiz.id,"question_count": old_quiz.question_count})

@login_required
def quiz_status(request, session_id):
    session = get_object_or_404(StudySession, id=session_id, user=request.user, goal_type="quiz")
    deck = session.quiz_list
    if not deck:
        return JsonResponse({"has_quiz": False})
    return JsonResponse({
        "has_quiz": True,
        "is_complete": deck.asked_count >= deck.question_count,
        "score": deck.score,
        "total": deck.question_count,
    })