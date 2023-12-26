from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix

import pickle
import gzip
import math
import numpy as np

NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}


class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [sum(NP_model.get(bit, 0)
                      for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,
                                                                            2).GetNonzeroElements()) / float(
            mol.GetNumAtoms()) if mol is not None else None
                  for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))

def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'diversity score': MolecularMetrics.diversity_scores(mols, data)}.items()}

    m1 = {'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1
