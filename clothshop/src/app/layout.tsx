import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
	title: 'Clothshop',
	description: 'A modern apparel storefront',
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body className={`${inter.className} min-h-screen flex flex-col`}>
				<header className="border-b sticky top-0 z-50 bg-white/80 backdrop-blur">
					<div className="container h-16 flex items-center justify-between">
						<Link href="/" className="font-bold text-xl">CLOTHSHOP</Link>
						<nav className="flex items-center gap-6 text-sm">
							<Link href="/men" className="hover:underline">Men</Link>
							<Link href="/women" className="hover:underline">Women</Link>
							<Link href="/kids" className="hover:underline">Kids</Link>
							<Link href="/search" className="hover:underline">Search</Link>
							<Link href="/cart" className="hover:underline">Cart</Link>
						</nav>
					</div>
				</header>

				<main className="flex-1">{children}</main>

				<footer className="border-t">
					<div className="container py-10 text-sm text-neutral-500">
						<p>© {new Date().getFullYear()} Clothshop. All rights reserved.</p>
					</div>
				</footer>
			</body>
		</html>
	);
}