"""Tools for directly working with a MatPES style DB."""

from __future__ import annotations

import functools
import os

import pandas as pd
from pymongo import MongoClient

from .data import get_data


@functools.cache
def _get_client(host: str, username: str | None, password: str | None) -> MongoClient:
    """Return a cached MongoClient.

    MongoClient is thread-safe and pools connections, so reusing it avoids paying the connection and authentication
    cost on every query.
    """
    return MongoClient(host=host, username=username, password=password, authSource="admin")


class MatPESDB:
    """A MatPES DB object. This requires access to a MatPES style DB. Typically meant for developers."""

    FUNCTIONALS = ("PBE", "r2SCAN")

    def __init__(self, dbname="matpes"):
        """
        Args:
            dbname (str): The name of the MatPES DB.
        """
        client = _get_client(
            os.environ.get("MATPES_HOST", "127.0.0.1"),
            os.environ.get("MATPES_USERNAME"),
            os.environ.get("MATPES_PASSWORD"),
        )
        self.db = client.get_database(dbname)

    def create_db(self):
        """
        Create a MatPES database from the json files.
        Note that any existing collections will be deleted.
        """
        for functional in self.FUNCTIONALS:
            data = get_data(functional=functional)
            coll = self.db.get_collection(functional.lower())
            coll.delete_many({})
            coll.insert_many(data)
            for field in [
                "matpes_id",
                "formula_pretty",
                "elements",
                "chemsys",
                "cohesive_energy_per_atom",
                "nsites",
                "nelements",
                "bandgap",
            ]:
                coll.create_index(field)

    def get_json(self, functional: str, criteria: dict) -> list:
        """
        Args:
            functional (str): The name of the functional to query.
            criteria (dict): The criteria to query.
        """
        return list(self.db.get_collection(functional.lower()).find(criteria))

    def get_df(self, functional: str, criteria=None, projection=None) -> pd.DataFrame:
        """
        Retrieve a pandas DataFrame from a MongoDB collection based on the provided
        criteria and projection.

        This method queries a MongoDB collection corresponding to the specified
        functional argument. It uses given criteria and projection to filter and
        retrieve the desired data, returning the results in the form of a pandas
        DataFrame. If no criteria or projection is provided, it uses default values.

        Parameters:
        functional: str
            The name of the collection to query, corresponding to a specific
            functional. The string is converted to lowercase.
        criteria: dict, optional
            A dictionary to filter the query results. Defaults to an empty
            dictionary if not provided.
        projection: list[str], optional
            A list of strings specifying the fields to include in the query
            results. Defaults to a predefined list of fields if not provided.

        Returns:
        pd.DataFrame
            A pandas DataFrame containing the retrieved data with the specified
            projection fields.

        Raises:
        None
        """
        collection = self.db.get_collection(functional.lower())
        criteria = criteria or {}
        projection = projection or [
            "matpes_id",
            "formula_pretty",
            "elements",
            "energy",
            "chemsys",
            "cohesive_energy_per_atom",
            "formation_energy_per_atom",
            "abs_forces",
            "nsites",
            "nelements",
            "bandgap",
        ]
        # A large batch size minimizes round trips to a remote DB. Passing columns ensures an empty result still
        # yields a DataFrame with the expected columns.
        docs = list(collection.find(criteria, projection=projection, batch_size=10000))
        return pd.DataFrame(docs, columns=projection)
