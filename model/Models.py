import z3

try:
    # Use Enums for categorical data (color in this case)
    Color, colors = z3.EnumSort(
        'Color', 
        ['red', 'blue', 'green']
    )
    red, blue, green = colors
except z3.Z3Exception:
    print('colors already defined')


class Model:
    def __init__(self):
        pass

    def __call__(self, i):
        # get a term from its index
        # In e.g., the boolean case, this is a proposition
        # in another case it could be an object
        return self.terms[i]

    def __str__(self):
        return '\n'.join([
            term.__str__()
            for term in self.terms
        ])


class Obj:
    def __init__(self, index, color):
        """
        - Each object is modelled as a dictionary of z3 variables, corresponding to the various properties of the object.
        - Features can be of different types e.g. Enumerate, binary, etc.
        """
        self.color = Const(f'{index}:color', color)
        self.price = Real(f'{index}:price')
        self.speed = Int(f'{index}:speed')

        self.terms = [
            self.color,
            self.price,
            self.speed
        ]

    def __getitem__(self, propname):
        return getattr(self, propname)

    def __str__(self):
        return f'Object: ' + ', '.join([
            f'{term}' for term in self.terms
        ])


class BooleanModel(Model):

    def __init__(self, n_props=2):

        self.terms = [
            z3.Bool(f'p{i}') 
            for i in range(n_props)
        ]

        Model.__init__(self)

class ObjectsModel(Model):

    def __init__(self, n_objects):

        self.objects = [
            Obj(index, Color)
            for index in range(n_objects)
        ]

        self.terms = []
        for obj in self.objects:
            self.terms.extend(obj.terms)

        Model.__init__(self)

    def __getitem__(self, i):
        # get one or more objects
        return self.objects[i]

    def get_prop(self, i, prop):
        # get a property of a specific object
        return self.objects[i][prop]

    def __len__(self):
        return len(self.objects)
