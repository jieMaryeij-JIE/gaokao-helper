from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI  # 新增DeepSeek需要的导入

# 加载环境变量
load_dotenv()

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化数据库
db = SQLAlchemy(app)

# 初始化DeepSeek客户端（替换原来的豆包客户端）
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


# 数据库模型定义（保持不变）
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    last_submission = db.Column(db.DateTime)


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(20), nullable=False)  # math/physics/biology/chemistry
    image_path = db.Column(db.String(200), nullable=False)
    ocr_result = db.Column(db.Text)
    submit_time = db.Column(db.DateTime, default=datetime.utcnow)


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    user_answer = db.Column(db.Text)
    correct_answer = db.Column(db.Text)
    is_correct = db.Column(db.Boolean)
    knowledge_points = db.Column(db.Text)
    explanation = db.Column(db.Text)
    difficulty = db.Column(db.String(10))  # easy/medium/hard


class WeeklyReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(20), nullable=False)
    report_content = db.Column(db.Text, nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    generate_time = db.Column(db.DateTime, default=datetime.utcnow)


# 创建数据库表
with app.app_context():
    db.create_all()
    # 创建一个默认用户（因为只给身边人用，不需要复杂的注册登录）
    if not User.query.filter_by(username='student').first():
        default_user = User(username='student')
        db.session.add(default_user)
        db.session.commit()


# 百度OCR工具函数（保持不变）
def get_baidu_access_token():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={os.getenv('BAIDU_API_KEY')}&client_secret={os.getenv('BAIDU_SECRET_KEY')}"
    response = requests.post(url)
    return response.json().get('access_token')


def ocr_image(image_path):
    access_token = get_baidu_access_token()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={access_token}"

    with open(image_path, 'rb') as f:
        image_data = f.read()

    files = {'image': image_data}
    response = requests.post(url, files=files)
    result = response.json()

    if 'words_result' in result:
        return '\n'.join([item['words'] for item in result['words_result']])
    else:
        return "OCR识别失败"


# DeepSeek API工具函数（完全替换原来的豆包函数）
def call_deepseek(prompt):
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system",
                 "content": "你是一位经验丰富的高中全科老师，擅长用通俗易懂的语言讲解知识点和错题解析。"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            reasoning_effort="high",  # 开启深度推理，提高解析质量
            extra_body={"thinking": {"type": "enabled"}}  # 开启思考过程，更适合教育场景
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI分析失败：{str(e)}"


# 路由定义
@app.route('/')
def index():
    user = User.query.first()
    today = datetime.utcnow().date()
    has_submitted_today = Submission.query.filter(
        Submission.user_id == user.id,
        db.func.date(Submission.submit_time) == today
    ).first() is not None

    return render_template('index.html', has_submitted_today=has_submitted_today)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 获取上传的文件
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # 保存文件
            filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 进行OCR识别
            ocr_result = ocr_image(file_path)

            # 保存提交记录
            user = User.query.first()
            submission = Submission(
                user_id=user.id,
                subject='unknown',  # 后面AI会自动分类
                image_path=file_path,
                ocr_result=ocr_result
            )
            db.session.add(submission)
            db.session.commit()

            # 跳转到AI分析页面
            return redirect(url_for('analyze', submission_id=submission.id))

    return render_template('upload.html')


@app.route('/analyze/<int:submission_id>')
def analyze(submission_id):
    submission = Submission.query.get_or_404(submission_id)

    # 设计提示词让AI分析题目（保持不变，只改了调用函数）
    prompt = f"""
    你是一位经验丰富的高中全科老师。请仔细分析以下卷子内容：

    {submission.ocr_result}

    请完成以下任务：
    1. 将卷子中的所有题目分割出来，每题单独列出
    2. 对每道题进行以下分析：
       a. 判断属于哪个学科（只能是数学、物理、生物、化学中的一个）
       b. 提取题目涉及的所有知识点
       c. 判断题目难度（易、中、难）
       d. 给出这道题的正确答案
       e. 如果用户在卷子上写了答案，请判断是否正确
       f. 如果做错了，给出详细的、高中生能听懂的解析

    3. 最后给出整份卷子的学科分类结果

    请严格按照以下JSON格式返回结果，不要添加任何其他内容：
    {{
        "subject": "整份卷子的主要学科",
        "questions": [
            {{
                "question_text": "题目内容",
                "user_answer": "用户写的答案，如果没写则为空字符串",
                "correct_answer": "正确答案",
                "is_correct": true/false,
                "knowledge_points": "知识点1, 知识点2, 知识点3",
                "explanation": "详细解析",
                "difficulty": "易/中/难"
            }},
            ...
        ]
    }}
    """

    # 调用DeepSeek API（替换原来的call_doubao）
    ai_result = call_deepseek(prompt)

    try:
        # 解析AI返回的JSON
        analysis = json.loads(ai_result)

        # 更新提交记录的学科
        submission.subject = analysis['subject']
        db.session.commit()

        # 保存所有题目
        for q in analysis['questions']:
            question = Question(
                submission_id=submission.id,
                question_text=q['question_text'],
                user_answer=q['user_answer'],
                correct_answer=q['correct_answer'],
                is_correct=q['is_correct'],
                knowledge_points=q['knowledge_points'],
                explanation=q['explanation'],
                difficulty=q['difficulty']
            )
            db.session.add(question)

        db.session.commit()

        # 跳转到分析结果页面
        return redirect(url_for('result', submission_id=submission.id))

    except Exception as e:
        return f"AI分析失败：{str(e)}<br><br>AI返回内容：{ai_result}"


@app.route('/result/<int:submission_id>')
def result(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    questions = Question.query.filter_by(submission_id=submission.id).all()

    return render_template('result.html', submission=submission, questions=questions)


@app.route('/api/check_submission')
def api_check_submission():
    user = User.query.first()
    today = datetime.utcnow().date()
    has_submitted = Submission.query.filter(
        Submission.user_id == user.id,
        db.func.date(Submission.submit_time) == today
    ).first() is not None

    return jsonify({'has_submitted': has_submitted})


@app.route('/generate_weekly_report')
def generate_weekly_report():
    user = User.query.first()

    # 计算本周的开始和结束日期
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    subjects = ['数学', '物理', '生物', '化学']

    for subject in subjects:
        # 获取本周该学科的所有错题
        questions = Question.query.join(Submission).filter(
            Submission.user_id == user.id,
            Submission.subject == subject,
            Question.is_correct == False,
            db.func.date(Submission.submit_time) >= start_of_week,
            db.func.date(Submission.submit_time) <= end_of_week
        ).all()

        if not questions:
            continue

        # 准备报告内容
        questions_text = '\n\n'.join([
            f"题目：{q.question_text}\n你的答案：{q.user_answer}\n正确答案：{q.correct_answer}\n知识点：{q.knowledge_points}\n解析：{q.explanation}"
            for q in questions
        ])

        # 让AI生成总结报告（调用DeepSeek）
        prompt = f"""
        你是一位经验丰富的高中{subject}老师。请根据以下本周错题，生成一份详细的学习总结报告：

        {questions_text}

        报告需要包含以下内容：
        1. 本周错题总数和正确率
        2. 高频错误知识点排名（按错误次数排序）
        3. 每个薄弱知识点的详细分析
        4. 针对性的下周学习建议

        报告语言要通俗易懂，适合高三学生阅读。
        """

        report_content = call_deepseek(prompt)

        # 保存报告
        report = WeeklyReport(
            user_id=user.id,
            subject=subject,
            report_content=report_content,
            start_date=start_of_week,
            end_date=end_of_week
        )
        db.session.add(report)

    db.session.commit()

    return redirect(url_for('reports'))


@app.route('/reports')
def reports():
    user = User.query.first()
    reports = WeeklyReport.query.filter_by(user_id=user.id).order_by(WeeklyReport.generate_time.desc()).all()

    return render_template('reports.html', reports=reports)


# 新增：历史记录页面路由
@app.route('/history')
def history():
    user = User.query.first()
    # 查询所有提交记录，按时间倒序排列
    submissions = Submission.query.filter_by(user_id=user.id).order_by(Submission.submit_time.desc()).all()

    return render_template('history.html', submissions=submissions)


# 新增：错题本页面路由
@app.route('/mistakes')
def mistakes():
    user = User.query.first()
    # 查询所有错题，按学科和时间倒序排列
    mistakes = Question.query.join(Submission).filter(
        Submission.user_id == user.id,
        Question.is_correct == False
    ).order_by(Submission.subject, Submission.submit_time.desc()).all()

    # 按学科分组
    mistakes_by_subject = {}
    for mistake in mistakes:
        subject = mistake.submission.subject
        if subject not in mistakes_by_subject:
            mistakes_by_subject[subject] = []
        mistakes_by_subject[subject].append(mistake)

    return render_template('mistakes.html', mistakes_by_subject=mistakes_by_subject)


# 运行应用
if __name__ == '__main__':
    app.run(debug=True)