# Legacy entrypoints

These files were the pre-package-layout scripts and artifacts. They are kept for reference only.
אנחנו אחרי revert של שני commits שגרמו regression ב-Inquiry autosave/list.

המצב עכשיו תקין:

* Autosave עובד
* פנייה חדשה שנשמרת אוטומטית מופיעה ב-/Inquiries
* כפתור "העתק" קיים רק ב-InquiryForm
* /Inquiries מציג כרגע "פתח" + "מחק"

עכשיו לבצע שינוי קטן בלבד:

להוסיף ליד כל פנייה ברשימת /Inquiries כפתור "העתק" ליד הכפתור "פתח".

## חשוב מאוד

לא לשנות:

* Autosave
* InquiryForm saveDraftNow
* invalidateQueries
* Inquiry.list('-created_date')
* סינון הרשימה
* מיון הרשימה
* delete logic
* עיצוב כפתור מחיקה
* schema
* Reminder Engine
* Dashboard

לא לחזור על התיקון הקודם שגרם regression.

## מה כן לעשות

ב-Inquiries.jsx בלבד, או במינימום קובץ עזר אם ממש חייבים:

1. בעמודת פעולות, ליד כפתור "פתח", להוסיף כפתור:
   "העתק"

הסדר הרצוי:
פתח | העתק | מחק

2. בלחיצה על "העתק":
   לבנות טקסט מתוך אותה רשומת inquiry שכבר קיימת ברשימה:

* שם לקוח
* סוג מבנה
* שטח
* טון קירור
* פירוט נוסף

3. להשתמש ב:
   navigator.clipboard.writeText(text)

4. אחרי הצלחה:
   לעדכן רק את אותה רשומה:
   copied_to_ai_at: new Date().toISOString()

5. להציג הודעה קצרה:
   "הפנייה הועתקה ללוח"

6. במקרה שגיאה:
   console.error('[Inquiries] failed to copy inquiry', error)
   alert("לא הצלחתי להעתיק את הפנייה")

## פורמט הטקסט

פנייה חדשה לניתוח:

שם לקוח: <client_name או "-">
סוג מבנה: <building_type או "-">
שטח: <area או "-">
טון קירור: <cooling_tons או "-">

פירוט נוסף:

<details או "-">

משימה:
נתח את הפנייה, זהה מידע חסר, וסכם אילו צעדים נדרשים להמשך טיפול.

## כללי בטיחות

1. לא לקרוא ל-AI בפועל.
2. לא לשלוח מידע ל-API חיצוני.
3. לא לשנות form_status.
4. לא לשנות submitted_at.
5. לא להפעיל Reminder Engine.
6. לא לשנות את query של Inquiries.
7. לא לשנות את delete button.
8. לא ליצור helper חדש שמשנה גם את InquiryForm, אלא אם זה ממש הכרחי. עדיף שינוי מקומי קטן ב-Inquiries.jsx בלבד.

## בדיקות חובה

1. לפתוח /InquiryForm בלי id.
2. להקליד שם לקוח.
3. לחכות ל-"נשמר".
4. לעבור ל-/Inquiries.
5. לוודא שהפנייה מופיעה כטיוטה.
6. ללחוץ "העתק" ליד אותה פנייה.
7. להדביק במקום אחר ולוודא שהטקסט תקין.
8. לוודא ש-copied_to_ai_at התעדכן.
9. לוודא ש"פתח" עדיין עובד.
10. לוודא ש"מחק" עדיין עובד עם confirm.
11. לוודא שלא נוצרו כפילויות.
12. להריץ npm run build.

בסוף לדווח:

* איזה קובץ נערך
* האם שינית רק את Inquiries.jsx
* האם copy עובד מהרשימה
* האם autosave עדיין מציג את הפנייה ברשימה
* האם build עבר

**Do not use for the current decision engine.** From the project root, use:

- `python -m src.log_triage.train`
- `python -m src.log_triage.predict`

If you run scripts here, run them from the **repository root** (`log_triage/`) with `PYTHONPATH=.` so `src.log_triage` resolves:

```bash
PYTHONPATH=. python legacy/train.py
PYTHONPATH=. python legacy/predict.py
```

Artifacts are written next to these scripts: `legacy/model.pkl`, `legacy/model_tfidf.pkl`.
