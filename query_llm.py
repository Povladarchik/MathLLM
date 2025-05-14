import requests
import logging

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class YandexGPTApiError(Exception):
    """Базовое исключение для ошибок, связанных с Yandex GPT API."""

    pass


class YandexGPTApi:
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, api_key: str, model_uri: str) -> None:
        """
        Инициализация клиента для взаимодействия с Yandex GPT API.

        :param api_key: API-ключ для авторизации.
        :param model_uri: URI модели в формате 'gpt://<folder_id>/yandexgpt/<model_version>'.
        """
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {api_key}",
        }
        self.model_uri = model_uri

    def send_prompt(
        self,
        system_text: str,
        user_text: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Отправляет запрос к модели Yandex GPT и возвращает сгенерированный ответ.

        :param system_text: Системное сообщение (инструкция для модели).
        :param user_text: Сообщение пользователя.
        :param temperature: Температура генерации (от 0 до 1). Чем выше — тем более творческий ответ.
        :param max_tokens: Максимальное количество токенов в ответе.
        :return: Текст ответа от модели.
        :raises YandexGPTApiError: Если произошла ошибка при выполнении запроса или получении ответа.
        """
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {"role": "system", "text": system_text},
                {"role": "user", "text": user_text},
            ],
        }

        try:
            logger.info("Отправка запроса к Yandex GPT API")
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при отправке запроса: {e}")
            raise YandexGPTApiError(f"Ошибка сети или API: {e}")

        try:
            result = response.json()
            logger.debug(f"Получен ответ от API: {result}")
            return result["result"]["alternatives"][0]["message"]["text"]
        except KeyError as e:
            logger.error(f"Ошибка разбора ответа API: отсутствует ключ {e}")
            raise YandexGPTApiError("Некорректный формат ответа от API") from e


if __name__ == "__main__":
    folder_id = ""
    secret_key = ""
    model_uri = f"gpt://{folder_id}/yandexgpt-lite"

    llm = YandexGPTApi(api_key=secret_key, model_uri=model_uri)
    print(
        llm.send_prompt(
            system_text="Ты умный ассистент!", user_text="Сколько тебе лет?"
        )
    )
