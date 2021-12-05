from telegram import Update
from telegram.ext import Updater, CommandHandler, PollAnswerHandler, MessageHandler, Filters, CallbackContext
from predictors.stock_predictor import StockPredictor
import pandas as pd
import json
import io
import warnings

warnings.filterwarnings('ignore')

class TelegramBot:

    def __init__(self):

        self.default_msg = 'Olá!\n\n' + \
            'Para solicitar uma previsão, escolha um dos comandos abaixo e selecione o ticker desejado:\n' + \
            'Previsão de ações: /predict_stocks\n' + \
            'Previsão de FIIs: /predict_reits\n' + \
            'Previsão de criptos: /predict_cryptos\n'

    def start(self, update: Update, context: CallbackContext) -> None:

        update.message.reply_text(
            self.default_msg
        )

    def retrieve_tickers(self, type):

        with open('./assets/available_tickers.json', 'r') as available_tickers_file:
            available_tickers = json.load(available_tickers_file)

        return available_tickers[type]

    def help_command(self, update: Update, context: CallbackContext):

        update.message.reply_text(
            self.default_msg
        )

    def default_message(self, update: Update, context: CallbackContext):

        update.message.reply_text(
            self.default_msg
        )

    def predict_from_ticker(self, ticker):

        predictor = StockPredictor(ticker=ticker, n_preds=90)
        predictor.get_data()
        predictor.get_tes_params()
        predictor.predict()
        predictor.build_plot_predictions()

        output = io.BytesIO()
        predictor.chart_pred.savefig(output, format='PNG')

        list_dates = predictor.future_dates[ : 7]
        list_preds = predictor.preds[ : 7]
        list_dates_and_preds = list(zip(list_dates, list_preds))

        return list_dates_and_preds, output


    def predict_stocks(self, update: Update, context: CallbackContext) -> None:

        questions = self.retrieve_tickers(type='stocks')

        message = context.bot.send_poll(
            update.effective_chat.id, 
            "Selecione uma ação", 
            questions, 
            is_anonymous=False, 
            allows_multiple_answers=False
        )

        payload = {
            message.poll.id: {
                'questions': questions, 
                'message_id': message.message_id, 
                'chat_id': update.effective_chat.id, 
                'answers': 0,
            }
        }

        context.bot_data.update(payload)

    def predict_reits(self, update: Update, context: CallbackContext) -> None:

        questions = self.retrieve_tickers(type='reits')

        message = context.bot.send_poll(
            update.effective_chat.id, 
            "Selecione uma ação", 
            questions, 
            is_anonymous=False, 
            allows_multiple_answers=False
        )

        payload = {
            message.poll.id: {
                'questions': questions, 
                'message_id': message.message_id, 
                'chat_id': update.effective_chat.id, 
                'answers': 0,
            }
        }

        context.bot_data.update(payload)

    def predict_cryptos(self, update: Update, context: CallbackContext) -> None:

        questions = self.retrieve_tickers(type='cryptos')

        message = context.bot.send_poll(
            update.effective_chat.id, 
            "Selecione uma ação", 
            questions, 
            is_anonymous=False, 
            allows_multiple_answers=False
        )

        payload = {
            message.poll.id: {
                'questions': questions, 
                'message_id': message.message_id, 
                'chat_id': update.effective_chat.id, 
                'answers': 0,
            }
        }

        context.bot_data.update(payload)

    def receive_poll_answer(self, update: Update, context: CallbackContext):

        answer = update.poll_answer
        poll_id = answer.poll_id

        print(answer)
        print(poll_id)

        try:
            questions = context.bot_data[poll_id]['questions']
        except KeyError:
            return

        context.bot.send_message(
            context.bot_data[poll_id]['chat_id'], 
            'Um momento, estamos calculando as previsões...'
        )

        selected_option = questions[answer.option_ids[0]]

        try:

            list_dates_and_preds, output = self.predict_from_ticker(selected_option)

            currency_symbol = 'U$' if 'USD' in selected_option else 'R$'

            context.bot.send_message(
                    context.bot_data[poll_id]['chat_id'], 
                    'Previsão para os próximos 7 dias úteis:\n\n' + \
                    '\n'.join([pd.to_datetime(i[0]).strftime('%d/%m/%Y') + ' -> ' + f'{currency_symbol}{i[1]:.2f}' for i in list_dates_and_preds])
                )

            context.bot.send_photo(context.bot_data[poll_id]['chat_id'], output.getvalue())

        except Exception as e:

            print(e)

            context.bot.send_message(
                context.bot_data[poll_id]['chat_id'], 
                'Desculpe, ocorreu um erro... tente novemente'
            )

        

    def main(self) -> None:

        arq = open('api_token.txt')
        api_token = arq.read()
        arq.close()

        updater = Updater(api_token)

        dispatcher = updater.dispatcher

        dispatcher.add_handler(CommandHandler('start', self.start))
        dispatcher.add_handler(CommandHandler('help', self.help_command))
        dispatcher.add_handler(CommandHandler('predict_stocks', self.predict_stocks))
        dispatcher.add_handler(CommandHandler('predict_reits', self.predict_reits))
        dispatcher.add_handler(CommandHandler('predict_cryptos', self.predict_cryptos))
        dispatcher.add_handler(PollAnswerHandler(self.receive_poll_answer))


        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.default_message))

        updater.start_polling()

        updater.idle()