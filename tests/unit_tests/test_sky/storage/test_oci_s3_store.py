"""Unit tests for OCI S3-compatible storage dispatch and store behavior."""

import unittest
from unittest import mock

from sky import cloud_stores
from sky import exceptions
from sky.data import storage as storage_lib


def _make_storage(**attrs):
    """Creates a bare Storage object without running __init__."""
    obj = object.__new__(storage_lib.Storage)
    obj.name = attrs.get('name', 'test-bucket')
    obj.source = attrs.get('source', None)
    obj.sync_on_reconstruction = attrs.get('sync_on_reconstruction', False)
    obj._bucket_sub_path = attrs.get('_bucket_sub_path', None)
    obj.stores = attrs.get('stores', {})
    obj._constructed = True
    obj._is_sky_managed = attrs.get('_is_sky_managed', None)
    return obj


class TestOciStoreDispatch(unittest.TestCase):
    """StoreType.OCI must dispatch on oci_s3.use_s3_api()."""

    def test_oci_s3_store_not_in_s3_compatible_registry(self):
        # Registering under 'OCI' would take over all StoreType.OCI dispatch
        # unconditionally (the registry is consulted before the StoreType.OCI
        # branches), bypassing the use_s3_api() gate.
        self.assertNotIn('OCI', storage_lib._S3_COMPATIBLE_STORES)

    def _run_from_metadata(self, use_s3_api: bool):
        metadata = storage_lib.AbstractStore.StoreMetadata(name='test-bucket',
                                                           source=None,
                                                           region=None,
                                                           is_sky_managed=True)
        sky_stores = {storage_lib.StoreType.OCI: metadata}
        storage_obj = _make_storage()

        native_sentinel = mock.Mock(name='native_store')
        s3_sentinel = mock.Mock(name='s3_store')
        with mock.patch.object(storage_lib.OciStore, 'from_metadata',
                               return_value=native_sentinel), \
             mock.patch.object(storage_lib.OciS3CompatibleStore,
                               'from_metadata', return_value=s3_sentinel), \
             mock.patch('sky.data.storage.oci_s3.use_s3_api',
                        return_value=use_s3_api), \
             mock.patch.object(storage_obj, '_add_store') as mock_add:
            storage_obj._add_store_from_metadata(sky_stores)

        mock_add.assert_called_once()
        return mock_add.call_args[0][0], native_sentinel, s3_sentinel

    def test_from_metadata_native_mode(self):
        store, native_sentinel, _ = self._run_from_metadata(use_s3_api=False)
        self.assertIs(store, native_sentinel)

    def test_from_metadata_s3_mode(self):
        store, _, s3_sentinel = self._run_from_metadata(use_s3_api=True)
        self.assertIs(store, s3_sentinel)

    def _run_add_store(self, use_s3_api: bool):
        storage_obj = _make_storage()
        native_cls = mock.Mock(name='OciStore')
        s3_cls = mock.Mock(name='OciS3CompatibleStore')
        with mock.patch('sky.data.storage.OciStore', native_cls), \
             mock.patch('sky.data.storage.OciS3CompatibleStore', s3_cls), \
             mock.patch('sky.data.storage.oci_s3.use_s3_api',
                        return_value=use_s3_api), \
             mock.patch.object(storage_obj, '_add_store'), \
             mock.patch.object(storage_obj, '_sync_store'):
            storage_obj.add_store(storage_lib.StoreType.OCI)
        return native_cls, s3_cls

    def test_add_store_native_mode(self):
        native_cls, s3_cls = self._run_add_store(use_s3_api=False)
        native_cls.assert_called_once()
        s3_cls.assert_not_called()

    def test_add_store_s3_mode(self):
        native_cls, s3_cls = self._run_add_store(use_s3_api=True)
        s3_cls.assert_called_once()
        native_cls.assert_not_called()


class TestOciS3CompatibleStore(unittest.TestCase):
    """OciS3CompatibleStore behavior that needs no cloud access."""

    def test_get_config(self):
        config = storage_lib.OciS3CompatibleStore.get_config()
        self.assertEqual(config.store_type, 'OCI')
        self.assertEqual(config.url_prefix, 'oci://')
        self.assertEqual(config.cloud_name, 'OCI')

    def test_region_suffix_in_name_rejected(self):
        with self.assertRaises(exceptions.StorageNameError) as context:
            storage_lib.OciS3CompatibleStore(name='bucket@us-sanjose-1',
                                             source=None)
        self.assertIn('S3-compatible', str(context.exception))

    def test_region_suffix_in_source_rejected(self):
        with self.assertRaises(exceptions.StorageNameError):
            storage_lib.OciS3CompatibleStore(name='bucket',
                                             source='oci://bucket@us-sanjose-1')

    def test_local_source_with_at_sign_accepted(self):
        # A local path containing '@' (e.g. run@v2) must not be rejected.
        # Only oci:// source URIs are inspected for the <bucket>@<region> form.
        # The region-suffix check runs before super().__init__; mock the
        # parent constructor so the test never validates credentials or
        # touches the cloud (the real initialize() creates missing buckets).
        with mock.patch.object(storage_lib.S3CompatibleStore,
                               '__init__',
                               return_value=None) as mock_parent_init:
            storage_lib.OciS3CompatibleStore(name='bucket',
                                             source='~/ckpts/run@v2')
        mock_parent_init.assert_called_once()

    def test_validate_name_follows_oci_rules(self):
        # OCI bucket names allow uppercase letters and underscores, which
        # the generic S3 naming rules would reject.
        name = 'My_Bucket.2024'
        self.assertEqual(storage_lib.OciS3CompatibleStore.validate_name(name),
                         name)


class TestOciCloudStorageDispatch(unittest.TestCase):
    """cloud_stores.get_storage_from_path must dispatch on use_s3_api()."""

    def test_native_mode(self):
        with mock.patch('sky.cloud_stores.oci_s3.use_s3_api',
                        return_value=False):
            store = cloud_stores.get_storage_from_path('oci://bucket/path')
        self.assertIsInstance(store, cloud_stores.OciCloudStorage)

    def test_s3_mode(self):
        with mock.patch('sky.cloud_stores.oci_s3.use_s3_api',
                        return_value=True):
            store = cloud_stores.get_storage_from_path('oci://bucket/path')
        self.assertIsInstance(store, cloud_stores.OciS3CloudStorage)


class TestOciS3CloudStorageCommands(unittest.TestCase):
    """Generated sync commands must use the AWS CLI with the OCI S3 files."""

    def test_make_sync_dir_command(self):
        cmd = cloud_stores.OciS3CloudStorage().make_sync_dir_command(
            'oci://bucket/path', '/dest')
        self.assertIn('s3://bucket/path', cmd)
        self.assertNotIn('oci://', cmd)
        self.assertIn('aws s3 sync', cmd)
        self.assertIn('AWS_SHARED_CREDENTIALS_FILE=~/.oci/s3.credentials', cmd)
        self.assertIn('AWS_CONFIG_FILE=~/.oci/s3.config', cmd)
        self.assertIn('--profile=oci', cmd)

    def test_make_sync_file_command(self):
        cmd = cloud_stores.OciS3CloudStorage().make_sync_file_command(
            'oci://bucket/path/file', '/dest')
        self.assertIn('s3://bucket/path/file', cmd)
        self.assertNotIn('oci://', cmd)
        self.assertIn('aws s3 cp', cmd)
        self.assertIn('AWS_SHARED_CREDENTIALS_FILE=~/.oci/s3.credentials', cmd)
        self.assertIn('--profile=oci', cmd)


class TestOciStoreDeleteGuard(unittest.TestCase):
    """Native ``OciStore.delete()`` must remove only the mounted sub-path for a
    user-provided (non-Sky-managed) bucket -- never the whole, user-owned
    bucket. This mirrors the guard every other store already has."""

    def _make_oci_store(self, *, sub_path, is_sky_managed):
        store = storage_lib.OciStore.__new__(storage_lib.OciStore)
        store.name = 'user-bucket'
        store.namespace = 'ns'
        store.region = 'us-sanjose-1'
        store._bucket_sub_path = sub_path  # pylint: disable=protected-access
        store.is_sky_managed = is_sky_managed
        return store

    def _run_delete(self, store):
        # Do NOT mask `with_oci_env`: it returns the real `&&`-joined shell
        # command, so the test exercises the actual command shape (the
        # sub-path delete must run that string through a shell, not split it).
        with mock.patch.object(storage_lib.subprocess,
                               'check_output',
                               return_value=b'') as mock_run:
            store.delete()
        cmd = mock_run.call_args[0][0]
        # The sub-path delete passes a shell string (shell=True); the
        # whole-bucket delete passes a split argv list.
        return cmd if isinstance(cmd, str) else ' '.join(cmd)

    def test_user_bucket_sub_path_deletes_only_prefix(self):
        store = self._make_oci_store(sub_path='run-123', is_sky_managed=False)
        cmd = self._run_delete(store)
        # Only the per-run prefix is removed; the bucket is left intact.
        self.assertIn('oci os object bulk-delete', cmd)
        self.assertIn('--prefix "run-123/"', cmd)
        self.assertIn('--bucket-name user-bucket', cmd)
        self.assertNotIn('oci os bucket delete', cmd)

    def test_sky_managed_bucket_deletes_whole_bucket(self):
        # A Sky-managed intermediate bucket is ours to delete entirely.
        store = self._make_oci_store(sub_path='run-123', is_sky_managed=True)
        cmd = self._run_delete(store)
        self.assertIn('oci os bucket delete', cmd)

    def test_no_sub_path_deletes_whole_bucket(self):
        store = self._make_oci_store(sub_path=None, is_sky_managed=False)
        cmd = self._run_delete(store)
        self.assertIn('oci os bucket delete', cmd)

    def test_sub_path_delete_runs_through_a_shell(self):
        # with_oci_env returns a '&&'-joined shell string (venv setup + source +
        # the oci call), so the sub-path delete must run it with shell=True, not
        # split it into argv.
        store = self._make_oci_store(sub_path='run-123', is_sky_managed=False)
        with mock.patch.object(storage_lib.subprocess,
                               'check_output',
                               return_value=b'') as mock_run:
            store.delete()
        self.assertIs(mock_run.call_args.kwargs.get('shell'), True)
        self.assertIsInstance(mock_run.call_args.args[0], str)


if __name__ == '__main__':
    unittest.main()
