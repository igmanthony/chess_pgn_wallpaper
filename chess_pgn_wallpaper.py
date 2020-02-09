import os
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.ndimage.filters import gaussian_filter


class Chessboard:
    '''A single chessboard on the wallpaper'''
    def __init__(
            self, position, board, piece_images, last_move, division,
            square, flipped
        ):
        self.division = division
        size = 8 * self.division
        self.x_start, self.y_start = position[0] * division, position[1] * division
        self.x_end, self.y_end = self.x_start + size, self.y_start + size
        self.board = board
        self.piece_images = piece_images
        self.last_move = last_move
        self.square = square
        self.flipped = flipped
        self.white, self.black, self.highlighted = self.gen_position_image()

    def gen_position_image(self):
        position = np.zeros((8 * self.division, 8 * self.division, 3))
        if self.last_move is not None:
            start_square = str_to_index(str(self.last_move)[:2], self.flipped) 
            end_square = str_to_index(str(self.last_move)[2:], self.flipped)
            for file, rank in [start_square, end_square]:
                xpix, ypix = rank * self.division, file * self.division
                position[
                    xpix: xpix + self.square, ypix: ypix + self.square, 2
                ] = self.move_circle(invert=square_is_light(file, rank))
        for square, piece in self.board.piece_map().items():
            file, rank = square_to_index(square, self.flipped)
            piece_image = self.piece_images[str(piece).lower()][:, :, 3]
            xpix, ypix = rank * self.division, file * self.division
            layer = 0 if str(piece).isupper() else 1 # white on 0, black on 1
            position[
                xpix: xpix + self.square, ypix: ypix + self.square, layer
            ] = piece_image
        return position[:, :, 0], position[:, :, 1], position[:, :, 2]

    def move_circle(self, invert = False):
        # Spagetti code for generating a pixelated circle with:
        # Outer and inner radii of rg and rg2 and x and y center coordinates of
        # a and b, respectively
        rg = (self.square / 2) - (self.division - self.square)
        rg2 = rg - ((self.division - self.square) / 1.5)
        a = int(rg) + (self.division - self.square)
        b = a
        yg, xg = np.ogrid[-a:self.square - a, -b:self.square-b]
        mask = xg*xg + yg*yg <= rg*rg
        array = np.ones((self.square, self.square))
        array[mask] = 255
        mask2 =  xg*xg + yg*yg <= rg2*rg2
        array[mask2] = 1
        blurred = gaussian_filter(array, sigma=.8)
        return blurred * -1 if invert else blurred


def main(directory, pieces_folder='pieces', pgns_folder='pgns'):
    width, height = 3840, 2160          # in pixels
    division_size, square_size = 32, 29 # in pixels
    piece_image_names = {
        'k': 'k.png',  # king
        'q': 'q.png',  # queen
        'b': 'b.png',  # bishop
        'n': 'n.png',  # night
        'r': 'r.png',  # rook
        'p': 'p.png',  # pawn
    } # one-letter abbreviations work better than full names for move lookups
    
    rows, cols = height // division_size, width // division_size

    # load images and pgn files
    piece_images = {
        name: imread(os.path.join(directory, pieces_folder, image_file))
        for name, image_file in piece_image_names.items()
    } # dictionary with piece name: piece image as key: value pair
    pgn_files = [
        file for file in os.listdir(os.path.join(directory, pgns_folder))
        if file.endswith('.pgn')
    ] # list of '.pgn'-ending files in the PGNS_FOLDER subdirectory

    # main loop through all miniature chess games; generates
    for pgn_file in pgn_files:
        with open(os.path.join(directory, pgns_folder, pgn_file), 'r') as pgn:
            game = chess.pgn.read_game(pgn)
        board = game.board()
        flipped = game.headers['Result'][0] != "1" # flip board if black won
        board_locations = gen_chessboard_locations(
            rows, cols, len([move for move in game.mainline_moves()]) + 1
        ) # where to put the boards in the main image / wallpaper

        chessboards = [
            Chessboard(
                board_locations[0], board, piece_images, None, division_size,
                square_size, flipped
            )
        ] # add the initial chessboard (no pieces moved)
        for move, location in zip(game.mainline_moves(), board_locations[1:]):
            board.push(move)
            chessboards.append(
                Chessboard(
                    location, board, piece_images, move, division_size,
                    square_size, flipped
                )
            )

        blank_chessboard = gen_blank_chessboard(square_size, division_size)
        wallpaper = gen_image_with_bright_center(width, height)
        wallpaper = add_random_grid(wallpaper, division_size, square_size)
        wallpaper = darken_grid_lines(
            wallpaper, division_size, square_size, amount=5
        )
        for cb in chessboards:
            foundation = wallpaper[cb.x_start: cb.x_end, cb.y_start: cb.y_end]
            wallpaper[cb.x_start: cb.x_end, cb.y_start: cb.y_end] = composite(
                blank_chessboard, 0.25, foundation, 1
            )
            wallpaper[cb.x_start: cb.x_end, cb.y_start: cb.y_end] = composite(
                cb.highlighted, 0.20, foundation, 1
            )
            wallpaper[cb.x_start: cb.x_end, cb.y_start: cb.y_end] = composite(
                255, cb.white / np.max(cb.white), foundation, 1 
            )
            wallpaper[cb.x_start: cb.x_end, cb.y_start: cb.y_end] = composite(
                25, cb.black / np.max(cb.black), foundation, 1
            )

        wallpaper = wallpaper - np.min(wallpaper)
        wallpaper = np.array(
            wallpaper / np.max(wallpaper) * 182 + 7, dtype=np.uint8
        )
        imwrite(os.path.join(directory, pgn_file[:-4] + '.png'), wallpaper)


# =========================== MATH & UTILITY FUNCTIONS ======================= #
def gaussian(size, b, c):
    '''Gaussian distribution with size: size, mu: b, sigma: c and height/a: 1'''
    return np.exp( -(((np.linspace(1, size, size) - b) / c)**2) )


def composite(top, top_alpha, bottom, bottom_alpha):
    numerator = (top * top_alpha) + bottom * bottom_alpha * (1 - top_alpha)
    denominator = top_alpha + bottom_alpha * (1 - top_alpha)
    return numerator / denominator

# =========================== WALLPAPER IMAGE FUNCTIONS ====================== #
def gen_image_with_bright_center(
        width, height, x_mu=(1/2.05), y_mu=(1/2.4), x_sig=(1/2.2), y_sig=1/2.5,
        i_min=0, i_max=255
    ):
    '''makes an image with a bright center (a 2-dimensional NumPy array)
    =======ARGUMENTS=======
    - width: the wallpaper width in pixels
    - height: the wallapepr height in pixels
    - x_mu: bright spot center x offset percent
    - y_mu: bright spot center y offset percent
    - x_sig: bright spot "sigma" value - percent of bright spot in x direction
    - y_sig: same as w_sig but for y direction
    - i_min: the bright spot/canvas minimum intensity (darkest areas)
    - i_max: the bright spot/canvas maximum intensity (lightest areas)
    '''
    horizontal_curve = gaussian(width, width * x_mu, width * x_sig)
    vertical_curve = gaussian(height, height * y_mu, height * y_sig)
    # NumPy matrix multiplication to generate the bright center surface
    surface = vertical_curve[:, None] @ horizontal_curve[:, None].T
    surface = surface / np.max(surface) * (i_max + 200 - i_min) + i_min
    # making the surface the correct light and darkness
    return surface // 3


def add_random_grid(image, division_size, square_size, noise=0.14):
    '''adds the random grid effect; increasing noise increases the variation in
    square brigness'''
    width, height = image.shape
    for y in range(0, height, division_size):
        for x in range(0, width, division_size):
            # get the average of all the pixels within the square
            mean = np.average(image[x: x + division_size, y: y + division_size])

            # shift those values by a random (normal distribution) amount
            shift = np.abs(np.random.normal(loc=1, scale=noise)) * (mean * 1.25)
            image[x: x + square_size, y: y + square_size] = np.around(shift)
    return image


def darken_grid_lines(image, division_size, square_size, amount=5):
    normalization_factor = np.max(image)
    width, height = image.shape
    for y in range(0, height, division_size):
        for x in range(0, width, division_size):
            image[x + square_size: x+division_size, y: y+square_size] -= amount
            image[x: x+division_size, y+square_size: y+division_size] -= amount
    return image


def gen_chessboard_locations(rows, cols, boardcount, sepx=2, sepy=2):
    '''Helper function for laying out the rows and columns of chessboards.
    Likely more complecated than it needs to be. Works from 1-70 boards or more
    ===Arguments===
    - rows: the number of square rows in the image (not pixels)
    - cols: the number of square columns in the image (not pixels)
    - boardcount: the total number of chessboards to lay out
    - sepx: the distance (in cols) between boards
    - sepy: the distance (in rows) between boards'''
    def layout_grid_position(squares, boards, separation, spacing):
        # assert maximum % 2 == 0
        coords = []
        previous = 0
        for i in range(boards):
            if i == 0:
                coord = squares // 2 - (spacing * ( boards // 2))
                if boards % 2 != 0:
                    coord -= spacing // 2
                coords.append(coord)
                previous = coord
            else:
                coord = previous + spacing
                coords.append(coord)
                previous = coord
        return coords
    # ensures even numbers with blank space on edges for maxes
    max_per_row = round(cols // (8 + sepy) / 2) * 2 - 2
    max_per_col = round(rows // (8 + sepx) / 2) * 2 - 2
    add_row = 0 if boardcount > 30 else 1
    number_of_rows = int(np.round((boardcount / max_per_row) + 0.1)) + add_row
    number_per_row = []
    for row in range(number_of_rows):
        if row == 0:
            number_per_row.append((boardcount) // number_of_rows)
        elif row <= ((boardcount) % number_of_rows):
            number_per_row.append(((boardcount) // number_of_rows) + 1)
        else:
            number_per_row.append((boardcount) // number_of_rows)
    assert sum(number_per_row) == boardcount
    coords = []
    y_coords = layout_grid_position(rows, number_of_rows, max_per_row, 8 + sepy)
    for i, row_count in enumerate(number_per_row):
        x_coords = layout_grid_position(cols, row_count, max_per_col, 8 + sepx)
        for x in x_coords:
            coords.append((y_coords[i] + 1, x + 1))
    return coords


def gen_blank_chessboard(square_size, division_size, white_i=300, black_i=50):
    blank_board = np.zeros([8 * division_size, 8 * division_size]) + 124
    for file in range(8):
        for rank in range(8):
            xpix, ypix = rank * division_size, file * division_size
            blank_board[
                xpix: xpix + square_size,
                ypix: ypix + square_size
            ] = white_i if square_is_light(rank, file) else black_i
    return blank_board


# =========================== CHESS FUNCTIONS ================================ #
def square_is_light(r, f):
    return (f % 2 == 0 and r % 2 == 0) or (f % 2 == 1 and r % 2 == 1)


def square_to_index(square_number, flipped):
    index = square_number % 8, abs(square_number - 63) // 8
    return index if not flipped else flip(*index)


def str_to_index(move, flipped):
    file_key = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    index = file_key[move[0]] - 1, 8 - int(move[1])
    return index if not flipped else flip(*index)


def flip(file, rank):
    return 7 - file, 7 - rank 


if __name__ == '__main__':
    main(os.path.dirname(os.path.abspath(__file__)))
