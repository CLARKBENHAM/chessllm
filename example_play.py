# %%
import threading
import multiprocessing
import time
import random
import math
import abc
from collections import namedtuple
import os
import itertools
import dotenv
from operator import itemgetter
from datetime import datetime
import csv
import re

import chess
import openai
from stockfish import Stockfish as _Stockfish

PATH = "/usr/local/bin/stockfish"

dotenv.load_dotenv()  # ".env", override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


# Stockfish implements this class
class APIEngine(abc.ABC):
    @abc.abstractmethod
    def get_best_move(self):
        "returns UCI string"
        pass

    @abc.abstractmethod
    def set_position(self, moves):
        pass

    @abc.abstractmethod
    def get_elo(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} named: {self.name}"


class Stockfish(_Stockfish, APIEngine):
    name = None

    def __init__(self, name=None, *vargs, **kwargs):
        _Stockfish.__init__(self, PATH, *vargs, **kwargs)
        self.name = name

    def __str__(self):
        elo = self.get_parameters()["UCI_Elo"]
        return super().__str__() + f" elo: {elo}"

    def get_elo(self):
        return self.get_parameters()["UCI_Elo"]

    # It can be faster to not use this because the games are shorter
    # was faster with 100ms than not
    # def get_best_move(self):
    #    """Limits time per move, may change elo.
    #    delete method if want raw elo
    #    """
    #    return _Stockfish.get_best_move_time(self, 250)


class OpenAI(APIEngine):
    name = None

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.san_moves = []
        self.uci_moves = []
        self.board = chess.Board()
        self.elo_est = 0  # metadata only

    def get_elo(self):
        return self.elo_est

    def __str__(self):
        return super().__str__() + f" model: {self.model}"

    def set_position(self, uci_moves):
        # Sticking with UCI moves for now, but prompt uses SAN
        self.uci_moves = uci_moves
        san_moves = []
        self.board.reset()
        for m in uci_moves:
            san_moves += [self.board.san(chess.Move.from_uci(m))]
            self.board.push_uci(m)
        self.san_moves = san_moves

    def _make_prompt(self):
        prompt = (
            #    """Complete the remainder of this game: `\n"""
            """[Event "FIDE World Championship match 2024"]\n"""
            """[Site "Los Angeles, USA"]\n"""
            """[Date "2024.11.11"]\n"""
            """[Round "13"]\n"""
            """[White "Carlsen, Magnus (NOR)"]\n"""
            """[Black "Nepomniachtchi, Ian (RUS)"]\n"""
            """[Result "0-1"]\n"""
            """[WhiteElo "2882"]\n"""
            """[White Title "GM"]\n"""
            """[BlackElo "2812"]\n"""
            """[BlackTitle "GM"]\n"""
            """[TimeControl "40/7200:20/3600:900+30"]\n"""
            """[UTCDate "2024.11.11"]\n"""
            """[UTCTime "12:00:00"]\n"""
            """[Variant "Standard"]\n"""
        )
        prompt = (
            '[Event "FIDE World Cup 2023"]\n[Site "Baku AZE"]\n[Date "2023.08.23"]\n[EventDate'
            ' "2021.07.30"]\n[Round "8.2"]\n[Result "1/2-1/2"]\n[White "Rameshbabu'
            ' Praggnanandhaa"]\n[Black "Magnus Carlsen"]\n[ECO "C48"]\n[WhiteElo "2690"]\n[BlackElo'
            ' "2835"]\n[PlyCount "60"]\n\n'
        )
        prompt += " ".join(
            [
                f"{i+1}.{wm} {bm}"
                for i, (wm, bm) in enumerate(
                    itertools.zip_longest(self.san_moves[0::2], self.san_moves[1::2], fillvalue="")
                )
            ]
        )

        if len(self.san_moves) % 2 == 0:
            prompt += f" {len(self.san_moves)//2+1}. " if len(self.san_moves) > 0 else "1. "
        return prompt

    def get_best_move(self):
        prompt = self._make_prompt()
        # print(f"prompt: `{prompt}`")
        suggestions = []
        i = 0
        errors = 0
        while i < 2 and errors < 3:
            try:
                response = openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=min(0.8 + (i / 8), 2),
                    max_tokens=6,  # longest san moves are 6 tokens: dxe8=R#
                    stop=[
                        ".",
                        "1-0",
                        "0-1",
                        "1/2",
                    ],  # sometimes space is the first character generated
                    n=5,  # prompt is 175 tokens, cheaper to get all suggestions at once
                )
            except Exception as e:
                print("API request error", e)
                time.sleep(3 + 2**errors)
                errors += 1
                continue
            # print(response)
            texts = list(
                set(dict.fromkeys(map(itemgetter("text"), response.choices)))
            )  # Dict's perserve order in py>=3.7
            print(f"OA responses: {texts}")
            for text in texts:
                san_move = text.strip().split(" ")[0].split("\n")[0].strip()
                try:
                    try:
                        uci_move = self.board.parse_san(san_move).uci()
                    except chess.AmbiguousMoveError as e:
                        print(f"WARN Ambigious '{san_move}'. Null contest? {e}")
                        color = chess.WHITE if len(self.uci_moves) % 2 == 0 else chess.BLACK
                        if len(san_move) == 2:
                            piece = chess.PAWN
                        else:
                            piece = next(
                                p
                                for p in [
                                    chess.PAWN,
                                    chess.KNIGHT,
                                    chess.BISHOP,
                                    chess.ROOK,
                                    chess.QUEEN,
                                    chess.KING,
                                ]
                                if chess.piece_symbol(p) == san_move[0].lower()
                            )
                        squares = [chess.square_name(p) for p in self.board.pieces(piece, color)]
                        uci_move = next(
                            (
                                f"{san_move[1:3]}{end}"
                                for end in squares
                                if f"{san_move[1:3]}{end}" in self.board.legal_moves
                            ),
                            None,
                        )
                        if uci_move is None:
                            uci_move = next(
                                m for m in self.board.legal_moves if m.uci()[:2] in squares
                            )
                    assert self.board.is_legal(self.board.parse_uci(uci_move))
                    return uci_move
                except Exception as e:
                    print(f"'{e}', '{text}', '{san_move}'")
                    suggestions += [text]
            i += 1
        return f"No valid suggestions: {'|'.join(suggestions)}"


def new_elos(elo1, elo2, result, k=24):
    """result in terms of player 1
    k: a factor to determine how much elo changes, higher changes more quickly
    """
    p = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (elo1 - elo2) / 400))
    elo1 += k * (result - p)
    elo2 += k * (p - result)
    return elo1, elo2


GameResults = namedtuple(
    "game_results",
    [
        "white",
        "black",
        "result",
        "time",
        "illegal_move",
        "moves",
        "fen",
    ],
)

StoreResults = namedtuple(
    "store_results",
    [
        *GameResults._fields,
        "white_elo",
        "black_elo",
        "eval",  # {'type': 'mate', 'value': nmoves_till_mate} or {'type': 'cp', 'value': centipawns_in_whites_favor}
    ],
)
# sf_elo = 1200
# oa_elo = 1200
# sf = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 512})
# sf2 = Stockfish("weak_stockfish", parameters={"Threads": 6, "Hash": 128})
# referee = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 1024, "Skill Level": 20})
# sf.set_elo_rating(sf_elo)
# sf2.set_elo_rating(sf_elo // 2)
# model = "gpt-3.5-turbo-instruct"
# oa = OpenAI(model, model)
# print(sf, sf2)

# moves = []
# for i in range(9):
#    sf.set_position(moves)
#    moves += [sf.get_best_move()]
# oa.set_position(moves)
# print(oa.get_best_move())


# %%
def engines_play(white, black, uci_moves=None):
    """2 engines play against each other, e1 is white, e2 black
    Args:
        white (APIEngine):
        black (APIEngine):
        uci_moves (list): optional list of starting uci strings, ["e2e4"]
    Returns results namedtuple where result is the value for white
    """
    board = chess.Board()

    if uci_moves is None:
        uci_moves = []
    for m in uci_moves:
        board.push_uci(m)
    white_first = len(uci_moves) % 2
    white.set_position(uci_moves)
    black.set_position(uci_moves)
    t = time.perf_counter()
    illegal_move = None
    for i in range(6000):  # max possible moves is 5.9k
        # print(board, "\n")
        turn = white if (i % 2) == white_first else black
        turn.set_position(uci_moves)
        m = turn.get_best_move()
        # print(m, i)
        try:
            board.push_uci(m)
        except Exception as e:
            print(e)
            illegal_move = m
        if m is None or illegal_move is not None:
            result = 0 if turn == white else 1
            illegal_move = m
            break
        uci_moves += [m]

        if board.outcome() is not None:
            result = board.outcome().result().split("-")[0]
            if result == "1/2":
                result = 0.5
            else:
                result = float(result)
            break
        elif board.can_claim_draw():  # outcome() uses 5 rep and 75 moves
            result = 0.5
            break

    t = time.perf_counter() - t
    return GameResults(
        white.name,
        black.name,
        result,
        t,
        illegal_move,
        uci_moves,
        board.fen(),
    )


def make_engines(sf_elo=1200, model="gpt-3.5-turbo-instruct"):
    sf = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 512, "UCI_Elo": sf_elo})
    oa = OpenAI(model, model)
    oa.elo_est = sf_elo
    return (sf, oa)


def play_threadsafe(elo, model):
    sf, oa = make_engines(elo, model)  # sf takes <1sec to init, but has locks to executable
    sf_white = bool(random.randint(0, 1))
    white, black = (sf, oa) if sf_white else (oa, sf)
    try:
        gr = engines_play(white, black)
    except Exception as e:
        print(e)
        return None

    print(chess.Board(gr.fen))
    eval = None
    if gr.illegal_move is not None:
        sf_elo = sf.get_parameters()["UCI_Elo"]
        sf.set_skill_level(20)
        sf.set_fen_position(gr.fen)
        eval = sf.get_evaluation()
        sf.set_elo_rating(sf_elo)
    r = StoreResults(*gr, white.get_elo(), black.get_elo(), eval)
    return tuple(r)


# sf, oa = make_engines()
# play_threadsafe(sf, oa)


# %%

if __name__ == "__main__":
    sf_elo = 1200
    oa_elo = 1200
    model = "gpt-3.5-turbo-instruct"
    all_results = []
    now = re.sub("\D+", "_", str(datetime.now()))
    NUM_CPU = 5
    NUM_GAMES = NUM_CPU * 10
    with open(f"results/results_{now}.csv", "a+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(StoreResults._fields)
        # engines = [make_engines() for _ in range(NUM_CPU)]
        # manager = multiprocessing.Manager()
        # engines = [(manager.Value(Stockfish, sf), oa) for sf, oa in engines]
        pool = multiprocessing.Pool(NUM_CPU)
        for _ in range(NUM_GAMES // NUM_CPU):
            # processes = []
            # for engine in engines:
            #    p = multiprocessing.Process(play_threadsafe, engine)
            #    print(p.args)  # print the arguments
            #    processes.append(p)
            #    p.start()
            # results = [p.join() for p in processes]

            results = [
                StoreResults(*r)
                for r in pool.starmap(play_threadsafe, [(sf_elo, model) for _ in range(NUM_CPU)])
            ]
            rsum = 0
            for r in results:
                if r is None:
                    print("skipped one")
                    continue
                writer.writerow(r)
                rsum += r.result
            all_results += [*results]
            sf_elo, oa_elo = new_elos(sf_elo, oa_elo, rsum)
            print(results, sf_elo, oa_elo)
            sf_elo = oa_elo
            # for sf, _ in engines:
            #    sf.set_elo_rating(sf_elo)
    print(all_results)
    print([(i.result, i.black_elo, i.eval) for i in all_results])
    # %%
