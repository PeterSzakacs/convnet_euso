import argparse


# custom argparse actions


def allowed_lengths(lengths=[]):
    class AllowedLengths(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) not in lengths:
                msg='argument "{f}" accepts {num} numbers of arguments left to right, got {l}'.format(
                    f=self.dest,num=lengths, l=len(values))
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return AllowedLengths

def required_length(nmin,nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength