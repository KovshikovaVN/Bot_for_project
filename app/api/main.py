from fastapi import FastAPI, HTTPException
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from uuid import UUID, uuid4
from typing import Optional
import os
import re
from app.core.logging import app_logger
from app.models import (
    GetMessageRequestModel,
    GetMessageResponseModel,
    IncomingMessage,
    Prediction
)

app = FastAPI()
GIGACHAT_API_KEY = '' #Your API
# Конфигурация GigaChat
#GIGACHAT_CREDENTIALS = os.getenv(response.text)  # Ваш токен из переменных окружения

llm = GigaChat(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)
payload = Chat(
    messages=[
        Messages(
            role=MessagesRole.SYSTEM,
            content="Ты самый обычный человек. Ты любишь поесть ночью, любишь футбол и сырные крекеры"
        )
    ],
    temperature=0.9,
    max_tokens=35,
)

class BotDetectionService:
    @staticmethod
    def calculate_text_anomalies(text: str) -> float:
        """Вычисляет показатель неестественности текста"""
        patterns = [
            (r'\b[А-ЯЁ]{4,}\b', 3.0),  # Капс
            (r'\b\d{6,}\b', 2.0),      # Длинные цифровые последовательности
            (r'[!?]{3,}', 1.5),        # Множественные знаки препинания
            (r'\b\w{20,}\b', 2.0),     # Очень длинные слова
            (r'[^\w\s]', 0.1)          # Спецсимволы
        ]
        
        score = 1.0
        for pattern, weight in patterns:
            if re.search(pattern, text):
                score *= (1 + weight)
        
        return min(1.0, score)

    @classmethod
    def analyze_message(cls, text: str) -> float:
        """Анализирует текст и возвращает вероятность бота"""
        try:
            # 1. Эвристический анализ
            anomaly_score = cls.calculate_text_anomalies(text)
            
            # 2. Запрос к GigaChat для оценки
            response = llm.chat(
                "Оцени вероятность (0-1), что этот текст написан ботом. "
                f"Ответь только числом:\n{text}"
            )
            llm_score = min(1.0, max(0.0, float(response.choices[0].message.content)))
            
            # 3. Комбинированная оценка
            return 0.7 * llm_score + 0.3 * anomaly_score
            
        except Exception as e:
            app_logger.error(f"Bot detection error: {str(e)}")
            return 0.5  # Значение по умолчанию при ошибке

@app.post("/predict", response_model=Prediction)
async def predict_message(msg: IncomingMessage):
    """Классификация сообщений с логированием"""
    app_logger.info(f"Analyzing message from dialog {msg.dialog_id}")
    
    probability = BotDetectionService.analyze_message(msg.text)
    app_logger.debug(f"Bot probability for message {msg.id}: {probability:.2f}")
    
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=round(probability, 4)
    )

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    payload.messages.append(Messages(role=MessagesRole.USER, content=body.last_msg_text))
    response = llm.chat(payload)
    payload.messages.append(response.choices[0].message)
    
    """messages = body.last_msg_text
        
        # 2. Отправляем запрос к LLM
    response = llm.chat(
            messages, max_tokens = 50,
        )
        
        # 3. Извлекаем и очищаем ответ"""
    llm_response = response.choices[0].message.content
    cleaned_response = llm_response.strip()
        # 4. Возвращаем в строгом формате для Streamlit
    return GetMessageResponseModel(
        new_msg_text=str(cleaned_response),
        dialog_id=body.dialog_id
    )
        
